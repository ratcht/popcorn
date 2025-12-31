import torch as t
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
  return triton.runtime.driver.active.get_current_target().backend == "cuda"  # type: ignore


@triton.jit
def _attn_fwd_inner(
	acc, l_i, m_i,
	Q_block, K_block_ptr, V_block_ptr,
	block_q_idx, softmax_scale,
	BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
	stage: tl.constexpr,
	offs_q: tl.constexpr, offs_kv: tl.constexpr,
	SEQ_LEN: tl.constexpr
):
	if stage == 1:
		# left of diagonal: from 0 to the left of the diag
		lo, hi = 0, block_q_idx * BLOCK_SIZE_Q
	elif stage == 2:
	  # diagonal: used only for the block where there is transition from unmasked and masked tokens (i.e diagonal)
		lo, hi = block_q_idx * BLOCK_SIZE_Q, (block_q_idx + 1) * BLOCK_SIZE_Q
		lo = tl.multiple_of(lo, BLOCK_SIZE_Q) # compiler hint
	else:
		lo, hi = 0, SEQ_LEN

	K_block_ptr = tl.advance(K_block_ptr, (0, lo))
	V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

	for start_kv in range(lo, hi, BLOCK_SIZE_KV):
		start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

		# compute QK
		K_block = tl.load(K_block_ptr)
		QK_block = tl.dot(Q_block, K_block)

		if stage == 2:
			# diagonal
			mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
			QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)  # mask out elems on diagonal
			m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
			QK_block -= m_ij[:, None]
		else:
			m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
			QK_block *= softmax_scale
			QK_block -= m_ij[:, None]

		# exp(qk_ij - m_ij)
		P_block = tl.math.exp(QK_block)

		# compute sum of exp (normalization factor of curr block)
		l_ij = tl.sum(P_block, axis=1)

		# correction factor exp(prev_max - curr_max)
		alpha = tl.math.exp(m_i - m_ij)

		# apply correction factor to prev l_i
		l_i = l_i * alpha + l_ij

		# load V
		V_block = tl.load(V_block_ptr)

		# f16
		P_block = P_block.to(tl.float16)

		# acc := O_block
		# computes O_new = (P x V) + (O_old * alpha)
		acc = acc * alpha[:, None]
		acc = tl.dot(P_block, V_block, acc=acc) # O_block += P_block @ V_block

		m_i = m_ij

		# move to next block of K, V
		K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K[head_dim, seq_len]
		V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V[seq_len, head_dim]


	return acc, l_i, m_i


@triton.autotune(
  configs=[
    triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_KV': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_KV': 128}, num_stages=3, num_warps=8),
  ],
  key=['SEQ_LEN', 'HEAD_DIM'],  # re-tune when these change
)
@triton.jit
def _attn_fwd(
	Q_ptr, K_ptr, V_ptr,
	softmax_scale,
	L_ptr, O_ptr,
	stride_Q_batch, stride_Q_head,stride_Q_seq, stride_Q_dim,
	stride_K_batch, stride_K_head,stride_K_seq, stride_K_dim,
	stride_V_batch, stride_V_head,stride_V_seq, stride_V_dim,
	stride_O_batch, stride_O_head,stride_O_seq, stride_O_dim,
	BATCH_SIZE: tl.constexpr, NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
	BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
	stage: tl.constexpr
):
	# tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

	# which block in the seq_len to process
	block_q_idx = tl.program_id(0) # block_index_q

	# which head and batch to process
	batch_head_idx = tl.program_id(1)
	batch_idx, head_idx = batch_head_idx // NUM_HEADS, batch_head_idx % NUM_HEADS

	# Q[batch_idx, head_idx, :, :], etc...

	# this gets (SEQ_LEN, HEAD_DIM) block in Q, K, V by indexing
	qkv_base = (
		batch_idx.to(tl.int64) * stride_Q_batch
		+ head_idx.to(tl.int64) * stride_Q_head
	)

	off_q = block_q_idx * BLOCK_SIZE_Q

	# Q[batch_idx, head_idx, block_q_idx*BLOCK_SIZE_Q:, :]
	Q_block_ptr = tl.make_block_ptr(
		base=Q_ptr + qkv_base,
		shape=(SEQ_LEN, HEAD_DIM),
		strides=(stride_Q_seq, stride_Q_dim),
		offsets=(off_q, 0),
		block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
		order=(1, 0)
	)

	# K[batch_idx, head_idx, :, :]
	# this is transposed seq_len and head_dim are flipped
	# we dont have an offset because we need all KV for a given Q
	K_block_ptr = tl.make_block_ptr(
		base=K_ptr + qkv_base,
		shape=(HEAD_DIM, SEQ_LEN),
		strides=(stride_K_dim, stride_K_seq),
		offsets=(0, 0),
		block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
		order=(1, 0)
	)

	# V[batch_idx, head_idx, :, :]
	V_block_ptr = tl.make_block_ptr(
		base=V_ptr + qkv_base,
		shape=(SEQ_LEN, HEAD_DIM),
		strides=(stride_V_seq, stride_V_dim),
		offsets=(0, 0),
		block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
		order=(1, 0)
	)

	# O[batch_idx, head_idx, block_q_idx * BLOCK_SIZE_Q:, :]
	O_block_ptr = tl.make_block_ptr(
		base=O_ptr + qkv_base,
		shape=(SEQ_LEN, HEAD_DIM),
		strides=(stride_O_seq, stride_O_dim),
		offsets=(off_q, 0),
		block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
		order=(1, 0)
	)

	# ====== END OF LINE 4 OF PSEUDOCODE REFERENCE ======

  # offs_q: offsets for the tokens in Q to process
	offs_q = off_q + tl.arange(0, BLOCK_SIZE_Q)

	# offs_kv: offsets for tokens in K, V to process
	# remember, we need all of KV so we dont have an offset within the block
	offs_kv = tl.arange(0, BLOCK_SIZE_KV)

	# m_i: running maximum. one for each query. init to -inf
	m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + float("-inf")

	# l_i: running sum. one for each query (as we sum attn scores by rows)
	l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

	# acc: the accumulator for the output, which is a group of rows of the O matrix
	O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

	Q_block = tl.load(Q_block_ptr)

	if stage == 1 or stage == 3:
		# step runs for non causal attention or for blocks to the left of the diag in causal attn
		O_block, l_i, m_i = _attn_fwd_inner(
			O_block, l_i, m_i,
			Q_block, K_block_ptr, V_block_ptr,
			block_q_idx, softmax_scale,
			BLOCK_SIZE_Q, BLOCK_SIZE_KV,
			4 - stage,  # tells us if we are on the left or right side of diag
			offs_q, offs_kv,
			SEQ_LEN
		)

	if stage == 3:
		# step runs for blocks to the right of the diag in non-causal attn
		O_block, l_i, m_i = _attn_fwd_inner(
			O_block, l_i, m_i,
			Q_block, K_block_ptr, V_block_ptr,
			block_q_idx, softmax_scale,
			BLOCK_SIZE_Q, BLOCK_SIZE_KV,
			2,
			offs_q, offs_kv,
			SEQ_LEN
		)

	m_i += tl.math.log(
		l_i
	) # used to compute logsumexp for backward pass

	O_block = O_block / l_i[:, None]
	l_ptrs = L_ptr + batch_head_idx * SEQ_LEN + offs_q

	tl.store(l_ptrs, m_i)
	tl.store(O_block_ptr, O_block.to(tl.float16))


class FlashAttention(t.autograd.Function):
	@staticmethod
	def forward(ctx, Q: t.Tensor, K: t.Tensor, V: t.Tensor, is_causal, softmax_scale):
		HEAD_DIM_K, HEAD_DIM_V = K.shape[-1], V.shape[-1]
		BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
		assert HEAD_DIM == HEAD_DIM_K == HEAD_DIM_V

		O = t.empty_like(Q)
		stage = 3 if is_causal else 1

		grid = lambda meta: (
			triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_Q"]), # num blocks of Q
			BATCH_SIZE * NUM_HEADS, # head of a specific batch
			1
		)

		# num_programs = BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q

		L = t.empty(
			(BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=t.float32
		) # logsumexp for the backward pass

		_attn_fwd[grid](
			Q_ptr=Q, K_ptr=K, V_ptr=V,
			softmax_scale=softmax_scale,
			L_ptr=L,
			O_ptr=O,
			stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1),stride_Q_seq=Q.stride(2), stride_Q_dim=Q.stride(3),
			stride_K_batch=K.stride(0), stride_K_head=K.stride(1),stride_K_seq=K.stride(2), stride_K_dim=K.stride(3),
			stride_V_batch=V.stride(0), stride_V_head=V.stride(1),stride_V_seq=V.stride(2), stride_V_dim=V.stride(3),
			stride_O_batch=O.stride(0), stride_O_head=O.stride(1),stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
			BATCH_SIZE=BATCH_SIZE, NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=HEAD_DIM, stage=stage
		)  # type: ignore[reportCallIssue]

		ctx.save_for_backward(Q, K, V, O, L)
		ctx.grid = grid
		ctx.softmax_scale = softmax_scale
		ctx.HEAD_DIM = HEAD_DIM
		ctx.causal = is_causal

		return O

def attention_torch(
	Q: t.Tensor, K: t.Tensor, V: t.Tensor, MASK: t.Tensor,
	softmax_scale: t.Tensor, is_causal=True,
):
	# reference impl
	P = t.matmul(Q, K.transpose(-2, -1)) * softmax_scale # QK^T / sqrt(HEAD_DIM)
	if is_causal:
		P[:, :, MASK==0] = float('-inf')
	P = t.softmax(P.float(), dim=-1).half()
	return t.matmul(P, V)


if __name__=="__main__":
	BATCH_SIZE = 32
	NUM_HEADS = 4
	SEQ_LEN = 256
	HEAD_DIM = 32
	is_causal = True
	dtype=t.float16

	print(f"device: {DEVICE}")
	Q = t.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
	K = t.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
	V = t.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
	MASK = t.tril(t.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))

	softmax_scale = 1 / (HEAD_DIM**0.5)
	dO = t.randn_like(Q) # needed for backward_pass

	torch_O = attention_torch(Q, K, V, MASK, softmax_scale, is_causal)
	# torch_O.backward(dO)
	# torch_dQ = Q.grad.clone()
	# torch_dK = K.grad.clone()
	# torch_dV = V.grad.clone()

	# Q.grad = None
	# K.grad = None
	# V.grad = None

	O = FlashAttention.apply(Q, K, V, is_causal, softmax_scale)
	# O.backward(dO)
	# dQ = Q.grad.clone()  # type: ignore[union-attr]
	# dK = K.grad.clone()  # type: ignore[union-attr]
	# dV = V.grad.clone()  # type: ignore[union-attr]

	print(f"torch: {torch_O}")
	print(f"triton: {O}")

	t.allclose(torch_O, O)

	print("Successful!")

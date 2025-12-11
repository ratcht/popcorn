NVCC = nvcc
NVCC_FLAGS = -O3
LIBS = -lcurand

BENCH_DIR = benchmarks
KERNEL_DIR = kernels
UTILS_DIR = utils

BENCHMARKS = bench_vector_add bench_matmul bench_conv1d bench_conv2d bench_reduction bench_softmax bench_sgemm bench_qkv_proj

.PHONY: all clean $(BENCHMARKS)

all: $(BENCHMARKS)

bench_vector_add: $(BENCH_DIR)/bench_vector_add.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS)

bench_matmul: $(BENCH_DIR)/bench_matmul.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS)

bench_sgemm: $(BENCH_DIR)/bench_sgemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS) -lcublas

bench_conv1d: $(BENCH_DIR)/bench_conv1d.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS)

bench_conv2d: $(BENCH_DIR)/bench_conv2d.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS)

bench_reduction: $(BENCH_DIR)/bench_reduction.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS)

bench_softmax: $(BENCH_DIR)/bench_softmax.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS)

bench_qkv_proj: $(BENCH_DIR)/bench_qkv_proj.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(BENCH_DIR)/$@ $(LIBS) -lcublas -lcublasLt

clean:
	rm -f $(BENCH_DIR)/bench_vector_add
	rm -f $(BENCH_DIR)/bench_matmul
	rm -f $(BENCH_DIR)/bench_conv1d
	rm -f $(BENCH_DIR)/bench_conv2d
	rm -f $(BENCH_DIR)/bench_reduction
	rm -f $(BENCH_DIR)/bench_softmax
	rm -f $(BENCH_DIR)/bench_sgemm
	rm -f $(BENCH_DIR)/bench_qkv_proj

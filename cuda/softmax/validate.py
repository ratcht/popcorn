import torch
import torch.nn.functional as F

# cuda params
ROWS = 1024
COLS = 1024
# =============================================

# load
input_t = torch.from_file("input.bin", size=ROWS * COLS, dtype=torch.float32)
output_t = torch.from_file("output.bin", size=ROWS * COLS, dtype=torch.float32)

# reshape
input_t = input_t.reshape(1, 1, ROWS, COLS).float()
output_t = output_t.reshape(1, 1, ROWS, COLS).float()

# torch ref
expected = F.softmax(input_t, dim=-1)

# compare
torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-7)
print("Passed!")

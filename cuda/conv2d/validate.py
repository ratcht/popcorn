import torch
import torch.nn.functional as F

# cuda params
INPUT_WIDTH = 8
INPUT_HEIGHT = 8
KERNEL_WIDTH = 2
KERNEL_HEIGHT = 2
PADDING = 1
# =============================================

OUTPUT_WIDTH = INPUT_WIDTH + 2 * PADDING - KERNEL_WIDTH + 1
OUTPUT_HEIGHT = INPUT_HEIGHT + 2 * PADDING - KERNEL_HEIGHT + 1

# load
input_t = torch.from_file("input.bin", size=INPUT_HEIGHT * INPUT_WIDTH, dtype=torch.int32)
kernel_t = torch.from_file("kernel.bin", size=KERNEL_HEIGHT * KERNEL_WIDTH, dtype=torch.int32)
gpu_output = torch.from_file("output.bin", size=OUTPUT_HEIGHT * OUTPUT_WIDTH, dtype=torch.int32)

# reshape
input_t = input_t.reshape(1, 1, INPUT_HEIGHT, INPUT_WIDTH).float()
kernel_t = kernel_t.reshape(1, 1, KERNEL_HEIGHT, KERNEL_WIDTH).float()
gpu_output = gpu_output.reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH)

# torch ref
expected = F.conv2d(input_t, kernel_t, padding=PADDING).squeeze().to(torch.int32)

# compare
torch.testing.assert_close(gpu_output, expected, rtol=0, atol=0)
print("Passed!")

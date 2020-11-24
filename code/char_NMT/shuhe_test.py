import torch
import torch.nn as nn
a = [[1, 2, 3], [2, 3, 4]]
x = torch.tensor(a, dtype=torch.float32)
print(x)
x = torch.unsqueeze(x, 0)
print(x)
conv = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3)
print(conv(x))
print(x)
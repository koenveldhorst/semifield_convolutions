import torch.nn.functional as F
import torch

data_im = torch.arange(1, 17).view(1, 1, 4, 4).float()

print(data_im)

unfolded = F.unfold(data_im, kernel_size=2, stride=2)
unfolded = unfolded.transpose(1,2)

print(unfolded)

out = torch.max(unfolded, dim=-1)[0]

print(out)
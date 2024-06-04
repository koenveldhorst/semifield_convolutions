import torch

# Create a tensor with requires_grad=True
x = torch.tensor([1, 6, 3, 5, 6], dtype=torch.float32, requires_grad=True)

# Perform a max operation along dimension 1
y = x.max(dim=0).values
y.backward()
print(y)
print(x.grad)

a = torch.tensor([1, 6, 3, 5, 7], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1, 7, 2, 4, 3], dtype=torch.float32, requires_grad=True)

# Perform a max operation element-wise
z = torch.max(a, b)
z = z.max()
z.backward()
print(z)  # This will print MaxBackward1
print(a.grad)

# Max with dim argument will use MaxBackward0 as the gradient function
# this will calculate the gradient towards the first max value

# Max without dim argument will use MaxBackward1 as the gradient function
# this will devide the gradient equally towards all max values
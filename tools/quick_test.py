# import torch


# temp = torch.tensor([
#     [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]],
#     [[-1, -2, -3, -4], [-5, -6, -7, -8], [0, 0, 0, 0]]
# ])

# print(temp[:, 0, 0])
# print(temp[:, 1, 1])
# print(temp[:, 2, 3])

# print(temp.shape)

# mask = torch.tensor([
#     [[1, 0, 0, 0], 
#     [0, 1, 0, 0], 
#     [0, 0, 0, 1]],
# ])
# mask = mask.expand(temp.shape).contiguous()

# temp_valid = temp[mask>0]

# print(temp_valid)
# print(temp_valid.shape)
# print(temp_valid.reshape(2, -1))
# print(temp_valid.reshape(2, -1).shape)


import torch

a = torch.tensor(
    [[1, 2],
    [3, 0]]
)



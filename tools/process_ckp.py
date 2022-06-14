import torch

ckp = torch.load("nfs/mobileAI2022/teacher/swin_l_w7_22k/epoch_200.pth")

print(ckp.keys())

ckp_processed = {}
for k,v in ckp['state_dict'].items():
    new_k = 'teacher_depther.' + k
    ckp_processed[new_k] = v
ckp['state_dict'] = ckp_processed

torch.save(ckp, "nfs/checkpoints/swinl_w7_22k_4x.pth")
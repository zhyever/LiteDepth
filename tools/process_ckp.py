import torch

ckp = torch.load("nfs/checkpoints/swinl_w7_22k_lightdecoder.pth")

print(ckp.keys())

ckp_processed = {}
for k,v in ckp['state_dict'].items():
    new_k = 'teacher_depther.' + k
    ckp_processed[new_k] = v
ckp['state_dict'] = ckp_processed

torch.save(ckp, "nfs/checkpoints/swinl_w7_22k_lightdecoder_prefix.pth")
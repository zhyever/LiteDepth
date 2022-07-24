import torch


model1 = 'nfs/mobileAI2022_final/distill_finetune/similarity_10/epoch_400.pth'

load_model1 = torch.load(model1)

new_param = {}

for key_item in load_model1['state_dict']:
    if 'student_depther' in key_item:
        val = load_model1['state_dict'][key_item] 
        new_key = key_item[16:]
        print(new_key)
        new_param[new_key] = val

load_model1['state_dict'] = new_param
torch.save(load_model1, 'nfs/mobileAI2022_final/final_model.pth')

import torch

checkpoint = torch.load('final_actor.pth', weights_only=False)
sd = checkpoint['state_dict']
for key in list(sd.keys()):
    print(key)
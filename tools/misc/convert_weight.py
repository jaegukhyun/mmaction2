import torch

# Convert Facebook official weight to mmaction style weight

weight = torch.load('/home/jaeguk/.cache/torch/hub/checkpoints/MViTv2_S_16x4_k400_f302660347.pyth', map_location='cpu')
model_state = weight['model_state']
new_model_state = {}
for key in model_state.keys():
    new_model_state['backbone.' + key] = model_state[key]
torch.save(new_model_state, '/home/jaeguk/.cache/torch/hub/checkpoints/MViTv2_S_16x4_k400_f302660347_mmaction2.pyth')

import torch
import sys

# Convert pytorch weight to mmaction style weight

weight_path = sys.argv[1]
weight = torch.load(weight_path, map_location='cpu')
try:
    model_state = weight['model_state']
except:
    model_state = weight
new_model_state = {}
for key in model_state.keys():
    new_model_state['backbone.' + key] = model_state[key]
new_weight_path = weight_path.split('.p')[0] + '_mmaction2.p' + weight_path.split('.p')[1]
print(weight_path, '==>', new_weight_path)
torch.save(new_model_state, new_weight_path)

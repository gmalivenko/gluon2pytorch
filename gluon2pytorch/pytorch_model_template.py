pytorch_model_template = """import torch
import torch.nn as nn
import torch.nn.functional as F

class {module_name}(nn.Module):
    def __init__(self):
        super({module_name}, self).__init__()
{inits}
    def forward(self, {inputs}):
{calls}
        return {outputs}

if __name__ == '__main__':
    net = {module_name}()
    net.load_state_dict(torch.load('{module_name_lower}.pt'))
    net.eval()
    print(net(torch.ones(1, 3, 224, 224)))
"""

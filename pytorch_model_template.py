pytorch_model_template = """
import torch
import torch.nn as nn

class {0}(nn.Module):
    def __init__(self):
        super({0}, self).__init__()
        {1}
    def forward(self, x):
        {2}
"""
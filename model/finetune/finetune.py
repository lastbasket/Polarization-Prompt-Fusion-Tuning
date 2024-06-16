"""
    CompletionFormer
    ======================================================================

    CompletionFormer implementation
"""

from model.completionformer.completionformer import CompletionFormer
import torch
import torch.nn as nn

class Finetune(nn.Module):
    def __init__(self, args):
        super(Finetune, self).__init__()

        self.model = CompletionFormer(args)
        self.model.load_state_dict(torch.load(args.pretrained_completionformer, map_location='cpu')['net'])

    def forward(self, sample):
    
        return self.model(sample)
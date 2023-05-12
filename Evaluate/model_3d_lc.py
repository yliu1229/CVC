import sys
sys.path.append('../Backbone')
from Backbone.uniformer import uniformer_select

import torch
import torch.nn as nn


class CVC_lc(nn.Module):
    def __init__(self, sample_size, network='uniformer_small', num_class=101, train_what='last'):
        super(CVC_lc, self).__init__()
        torch.cuda.manual_seed(233)
        self.sample_size = sample_size
        self.num_class = num_class
        self.train_what = train_what

        print('=> Using Pretrain + FC model ')
        self.backbone, self.feature_dim = uniformer_select(network, img_size=sample_size)

        # linear probe
        self.linear_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.num_class)
        )
        self._initialize_weights(self.linear_classifier)

    def forward(self, block):
        # block: [B, Num of frames, C, H, W]
        (B, T, C, H, W) = block.shape
        block = block.view(B, C, T, H, W)

        enable_grad = self.train_what != 'last'
        with torch.set_grad_enabled(enable_grad):
            feature = self.backbone(block)  # feature = (B, feature_size(512))
        del block

        out = self.linear_classifier(feature)

        return out

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)


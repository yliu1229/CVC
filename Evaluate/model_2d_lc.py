import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../Backbone')
from Backbone.vision_transformer import vit_select


class CVC_2d_lc(nn.Module):

    def __init__(self, sample_size=128, network='vit_small', num_class=200, train_what='last'):
        super(CVC_2d_lc, self).__init__()
        torch.cuda.manual_seed(233)
        self.train_what = train_what

        print('Using pretrained CVC with image + FC model')
        self.backbone, self.feature_dim = vit_select(network, img_size=[sample_size, sample_size])

        # linear probe
        self.linear_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, num_class)
        )

        self._initialize_weights(self.linear_classifier)

    def forward(self, image, image_aug=None):
        # image of [B, C, H, W]
        # Extract feature
        enable_grad = self.train_what != 'last'
        with torch.set_grad_enabled(enable_grad):
            feature = self.backbone(image)  # feature = (B, feature_size(512))
        del image

        out = self.linear_classifier(feature)

        return out

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)


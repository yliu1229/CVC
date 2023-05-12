import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../Backbone')
from Backbone.uniformer import uniformer_select


class CVC_3d(nn.Module):

    def __init__(self, sample_size=128,
                 network='uniformer_small',
                 representation_dim=512,
                 cluster_num=400,
                 with_aug=False):
        super(CVC_3d, self).__init__()
        torch.cuda.manual_seed(233)

        print('Using CVC with video learning module...')
        self.with_aug = with_aug

        self.backbone, self.feature_dim = uniformer_select(network, img_size=sample_size)

        # Instance projector
        self.instance_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, representation_dim),
        )

        # Cluster projector
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.feature_dim, cluster_num),
            nn.Softmax(dim=1)
        )

        self._initialize_weights(self.instance_projector)
        self._initialize_weights(self.cluster_projector)

    def forward(self, block, block_aug=None):
        # block: [B, Num of frames, C, H, W]
        # Extract feature
        (B, T, C, H, W) = block.shape
        block = block.view(B, C, T, H, W)
        feature = self.backbone(block)  # feature = [B, feature_dim(512)]
        del block
        # Get instance feature matrix
        instance_m = self.instance_projector(feature)

        instance_aug_m = None
        if self.with_aug:
            block_aug = block_aug.view(B, C, T, int(H/2), int(W/2))
            feature_aug = self.backbone(block_aug)
            instance_aug_m = self.instance_projector(feature_aug)
            # instance_aug_m = F.normalize(instance_aug_m, dim=1)
            del block_aug, feature_aug

        # Get cluster feature matrix
        cluster_m = self.cluster_projector(feature)  # matrix = [B, cluster_num(400)]
        del feature

        return [instance_m, cluster_m, instance_aug_m]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)


if __name__ == '__main__':
    model = CVC_3d(with_aug=True)

    input = torch.randn(4, 8, 3, 64, 64)
    instance, cluster, instance_aug = model(input, input)
    print(instance.shape, cluster.shape, instance_aug.shape)

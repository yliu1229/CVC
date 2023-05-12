import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../Backbone')
from Backbone.vision_transformer import vit_select


class CVC_2d(nn.Module):

    def __init__(self, sample_size=128,
                 network='vit_small',
                 representation_dim=512,
                 cluster_num=200,
                 with_aug=False):
        super(CVC_2d, self).__init__()
        torch.cuda.manual_seed(233)

        print('Using CVC with image learning module...')
        self.with_aug = with_aug
        self.cluster_num = cluster_num
        self.representation_dim = representation_dim

        self.backbone, self.feature_dim = vit_select(network, img_size=[sample_size, sample_size])

        # Instance projector
        if self.representation_dim is not None:
            self.instance_projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, representation_dim),
            )
            self._initialize_weights(self.instance_projector)

        # Cluster projector
        if self.cluster_num is not None:
            self.cluster_projector = nn.Sequential(
                nn.Linear(self.feature_dim, cluster_num),
                nn.Softmax(dim=1)
            )
            self._initialize_weights(self.cluster_projector)

    def forward(self, image, image_aug=None):
        # image of [B, C, H, W]
        # Extract feature
        (B, C, H, W) = image.shape
        feature = self.backbone(image)  # feature = [B, feature_dim(384)]
        del image
        # Get instance feature matrix
        if self.representation_dim is not None:
            feature = self.instance_projector(feature)

        instance_aug_m = None
        if self.with_aug:
            feature_aug = self.backbone(image_aug)
            instance_aug_m = self.instance_projector(feature_aug)
            del image_aug, feature_aug

        # Get cluster feature matrix
        cluster_m = None
        if self.cluster_num is not None:
            cluster_m = self.cluster_projector(feature)  # matrix = [B, cluster_num(400)]

        return [feature, cluster_m, instance_aug_m]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)


if __name__ == '__main__':
    model = CVC_2d(with_aug=True)

    input = torch.randn(5, 3, 64, 64)
    instance, cluster, instance_aug = model(input, input)
    print(instance.shape, cluster.shape, instance_aug.shape)

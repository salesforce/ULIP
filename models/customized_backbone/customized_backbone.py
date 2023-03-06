import torch.nn as nn
import torch.nn.functional as F

class CUSTOMIZED_BACKBONE(nn.Module):
    """
    This is a template for defining your customized 3D backbone and use it for pre-training in ULIP framework.
    The expected input is Batch_size x num_points x 3, and the expected output is Batch_size x point_cloud_feat_dim
    """
    def __init__(self):
        pass

    def forward(self, xyz):
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CONV4(MetaModule):
    def __init__(self, in_channels, hidden_size, fc_in, rot=False, pretrain=False, contra=False, non_local=False):
        super(CONV4, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.rot = rot
        self.pretrain = pretrain
        self.contra = contra
        self.non_local = non_local

        self.encoder = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )
        if self.non_local:
            self.non_local_head = MetaNonLocalBlock(hidden_size)
        if self.rot:
            self.rot_head = MetaLinear(fc_in, 4)
        if self.pretrain:
            self.pretrain_head = nn.Linear(fc_in, 64)
        if self.contra:
            self.contra_head = nn.Linear(fc_in, 128)
        self.linear = MetaLinear(fc_in, 2)


    def forward(self, x, params=None, ret_emb=False):
        embeddings = self.encoder(x, params=self.get_subdict(params, 'encoder'))
        embeddings = embeddings
        if self.non_local:
            embeddings = self.non_local_head(embeddings, params=self.get_subdict(params, 'non_local_head'))
        embeddings = embeddings.view((embeddings.size(0), -1))
        if ret_emb:
            return embeddings
        if self.pretrain:
            c = self.pretrain_head(embeddings)
            if self.contra:
                feature = self.contra_head(embeddings)
                return c, F.normalize(feature, dim=1)
            return c
        out = self.linear(embeddings, params=self.get_subdict(params, 'linear'))
        if self.rot:
            rotation = self.rot_head(embeddings, params=self.get_subdict(params, 'rot_head'))
            return out, rotation
        return out, embeddings


class MetaNonLocalBlock(MetaModule):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(MetaNonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # Convolution layers for theta, phi, and g
        self.conv_theta = MetaConv2d(in_channels, self.inter_channels, kernel_size=1)        
        self.phi = MetaSequential(
            MetaConv2d(in_channels, self.inter_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.g = MetaSequential(
            MetaConv2d(in_channels, self.inter_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_z = MetaConv2d(self.inter_channels, in_channels, kernel_size=1)
            
    def forward(self, x, params=None):
        batch_size = x.size(0)
        H, W = x.size()[2:]
        
        # Theta path: (B, C', H, W) -> (B, H*W, C')
        theta = self.conv_theta(x, params=self.get_subdict(params, 'conv_theta'))
        theta = theta.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        
        # Phi path: (B, C', H', W') -> (B, C', H'*W')
        phi = self.phi(x, params=self.get_subdict(params, 'phi'))
        phi = phi.view(batch_size, self.inter_channels, -1)
        
        # Affinity matrix: (B, H*W, H'*W')
        f = torch.matmul(theta, phi)
        f = f.softmax(dim=-1)
        
        # G path: (B, C', H'*W') -> (B, H'*W', C')
        g = self.g(x, params=self.get_subdict(params, 'g')).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        
        # Compute output: (B, H*W, C') -> (B, C', H, W)
        y = torch.matmul(f, g).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        
        # Project channels and add residual
        z = self.conv_z(y, params=self.get_subdict(params, 'conv_z'))
        return z + x

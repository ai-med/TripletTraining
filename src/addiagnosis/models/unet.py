# This file is part of From Barlow Twins to Triplet Training: Differentiating Dementia with Limited Data (Triplet Training).
#
# Triplet Training is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Triplet Training is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Triplet Training. If not, see <https://www.gnu.org/licenses/>.

import monai
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ConvBnReLU, Decoder_ResBlock, Upsample
from .resnet import BaseResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

class Decoder_BaseResNet(nn.Module):
    def __init__(self, 
                encoder_in_channels: int, 
                n_blocks: int = 4, 
                bn_momentum: float = 0.05, 
                n_basefilters: int = 8, 
                resnet_version: str = 'base', 
                bn_track_running_stats: bool = True,
                remain_downsample_steps: int = None,
                center: bool = True):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.encoder_in_channels = encoder_in_channels

        head_channels = n_basefilters * (2 ** (n_blocks - 1))

        if center:           
            self.center = CenterBlock(head_channels, head_channels)
        else:
            self.center = nn.Identity()
        
        self.bn1 = nn.BatchNorm3d(head_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")
               
        blocks = []
        n_filters = n_basefilters * (2 ** (n_blocks - 1))//2
        for i in range(n_blocks):
            if i < (n_blocks - 1):           
                blocks.append(Decoder_ResBlock(n_filters, n_filters, n_filters//2))            
                n_filters /= 2 
            else:
                n_filters *= 2
                blocks.append(Decoder_ResBlock(n_filters, n_filters, n_filters))
            n_filters = int(n_filters)

        self.blocks = nn.ModuleList(blocks)

        self.conv2 = nn.Conv3d(in_channels=n_filters, out_channels=encoder_in_channels, kernel_size=1, stride=1, padding = 1)
        self.bn2 = nn.BatchNorm3d(encoder_in_channels)

        self.resnet_version = resnet_version


    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder   
        # remove the global average output
        head = features[1]
        skips = features[2:]
        skips.pop(-2)

        x = self.center(head)
        x = self.bn1(x)
        x = self.relu1(x)
        x = Upsample(x, skips[0].shape[-1], skips[0].shape[1])
        
        if self.resnet_version == 'base':
            for i, decoder_block in enumerate(self.blocks):
                skip = skips[i] if i < len(skips) else None
                x = decoder_block(x, skip)

            x = self.conv2(x)
            x = self.bn2(x)
                
        else:
            raise NotImplementedError('Have not implemented other type of resnet decoder')
        return x
        

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = ConvBnReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        conv2 = ConvBnReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        super().__init__(conv1, conv2)

class Unet_ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_blocks: int = 4,
        bn_momentum: float = 0.05,
        n_basefilters: int = 16,
        dropout_rate: float = 0.2,
        resnet_version: str = 'base',
        bn_track_running_stats: bool = True,
        output_features: bool = False,
        remain_downsample_steps: int = None,
        center_block: bool = True,
    ):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.encoder = BaseResNet(
            in_channels=in_channels,
            n_blocks=n_blocks,
            bn_momentum=bn_momentum,
            n_basefilters=n_basefilters,
            resnet_version=resnet_version,
            bn_track_running_stats = bn_track_running_stats,
            remain_downsample_steps = remain_downsample_steps
        )

        self.decoder = Decoder_BaseResNet(
            encoder_in_channels = in_channels, 
            n_blocks=n_blocks, 
            bn_momentum=bn_momentum, 
            n_basefilters=n_basefilters, 
            resnet_version=resnet_version, 
            bn_track_running_stats = bn_track_running_stats,
            remain_downsample_steps = remain_downsample_steps,
            center = center_block
        )
    
    def forward(self, x):
        out, features = self.encoder(x, get_all_features = True)
        restored_x = self.decoder(features)

        return restored_x, out, features





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
from .blocks import ConvBnReLU, ResBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseResNet(nn.Module):
    def __init__(self, 
                in_channels: int, 
                n_outputs: int = 3, 
                n_blocks: int = 6, 
                bn_momentum: float = 0.05, 
                n_basefilters: int = 16, 
                resnet_version: str = 'base', 
                bn_track_running_stats: bool = True,
                remain_downsample_steps: int = None,
                no_downsample: bool = False):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.in_channels = in_channels
        self.output_num = n_outputs
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum, kernel_size=5, bn_track_running_stats = self.bn_track_running_stats)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.no_downsample = no_downsample

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        blocks = [
            ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum, 
                    bn_track_running_stats = self.bn_track_running_stats, no_downsample = self.no_downsample)
        ]
        n_filters = n_basefilters
        for i in range(n_blocks - 1):
            if remain_downsample_steps and i > remain_downsample_steps:            
                blocks.append(ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=1, 
                bn_track_running_stats = self.bn_track_running_stats, no_downsample = self.no_downsample))
            else:
                blocks.append(ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=2, 
                bn_track_running_stats = self.bn_track_running_stats, no_downsample = self.no_downsample))
            n_filters *= 2
        
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.resnet_version = resnet_version
        self.resnets = { 'base': None,
            'resnet18': monai.networks.nets.resnet18, 
            'resnet34': monai.networks.nets.resnet34,
            'resnet50': monai.networks.nets.resnet50, 
            'resnet101': monai.networks.nets.resnet101,
            'resnet152': monai.networks.nets.resnet152
        }
        assert resnet_version in self.resnets

    def forward(self, x, get_all_features = False):
        features = []
        if self.resnet_version == 'base':
            out = self.conv1(x)
            features.append(out)
            out = self.pool1(out)

            for block in self.blocks:
                if get_all_features:
                    features.append(out)
                out = block(out)

            if get_all_features:
                features.append(out)

            out = self.global_pool(out)
            if get_all_features:
                features.append(out)

            out = out.view(out.size(0), -1) 
            if get_all_features:
                return out, features
            else:
                return out
        
        elif self.resnet_version != 'base':

            # resnet_model = self.resnets[self.resnet_version](n_input_channels = self.in_channels, num_classes = self.output_num, feed_forward = False, pretrained=False).to(device)
            resnet_model = self.resnets[self.resnet_version].to(device)
            return resnet_model(x)

class SingleResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int = 3,
        n_blocks: int = 6,
        bn_momentum: float = 0.05,
        n_basefilters: int = 16,
        dropout_rate: float = 0.2,
        resnet_version: str = 'base',
        bn_track_running_stats: bool = True,
        output_features: bool = False,
        remain_downsample_steps: int = None,
        num_mlp: int = 2,
        no_downsample: bool = False,
    ):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.resnet = BaseResNet(
            in_channels=in_channels,
            n_outputs=n_outputs,
            n_blocks=n_blocks,
            bn_momentum=bn_momentum,
            n_basefilters=n_basefilters,
            resnet_version=resnet_version,
            bn_track_running_stats = self.bn_track_running_stats,
            remain_downsample_steps = remain_downsample_steps,
            no_downsample = no_downsample,
        )
        self.n_outputs = n_outputs
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.output_features = output_features
        self.num_mlp = num_mlp

        n_filters_out = n_basefilters * (2 ** (n_blocks - 1))
        if resnet_version == 'base':
            if num_mlp == 2:
                self.fc1 = nn.Linear(n_filters_out, n_filters_out, bias=False)
                self.bn = nn.BatchNorm1d(n_filters_out, track_running_stats = self.bn_track_running_stats)
                self.relu = nn.ReLU(inplace=True)

                n_outputs = 1 if n_outputs == 2 else n_outputs
                self.fc2 = nn.Linear(n_filters_out, n_outputs)
            
            elif num_mlp == 3:
                self.fc1 = nn.Linear(n_filters_out, n_filters_out, bias=False)
                self.bn1 = nn.BatchNorm1d(n_filters_out, track_running_stats = self.bn_track_running_stats)
                self.relu1 = nn.ReLU(inplace=True)
                self.fc2 = nn.Linear(n_filters_out, n_filters_out // 2, bias=True)
                self.bn2 = nn.BatchNorm1d(n_filters_out // 2, track_running_stats = self.bn_track_running_stats)
                self.relu2 = nn.ReLU(inplace=True)

                self.fc3 = nn.Linear(n_filters_out // 2, n_outputs)

        else:
            self.fc1 = nn.Linear(512, n_filters_out, bias=False)
            self.bn = nn.BatchNorm1d(n_filters_out, track_running_stats = self.bn_track_running_stats)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(n_filters_out, n_outputs)


    def forward(self, inputs, get_features: bool = False, get_all_features: bool = False):
        if get_features is None:
            out = self.resnet(inputs)
            features = out
            out = self.dropout(out)
            if self.num_mlp == 2:
                out = self.fc1(out)
                out = self.relu(self.bn(out))
                out = self.fc2(out)
            elif self.num_mlp == 3:
                out = self.fc1(out)
                out = self.relu1(self.bn1(out))
                out = self.fc2(out)
                out = self.relu2(self.bn2(out))
                out = self.fc3(out)
            return features, out
        else:
            if get_features and get_all_features:
                out, features = self.resnet(inputs, get_all_features = get_all_features)
                return out, features
            else: 
                out = self.resnet(inputs)
    
            if not get_features:
                out = self.dropout(out)
                if self.num_mlp == 2:
                    out = self.fc1(out)
                    out = self.relu(self.bn(out))
                    out = self.fc2(out)
                elif self.num_mlp == 3:
                    out = self.fc1(out)
                    out = self.relu1(self.bn1(out))
                    out = self.fc2(out)
                    out = self.relu2(self.bn2(out))
                    out = self.fc3(out)
                # if self.n_outputs == 2:
                #     out = torch.sigmoid(out)
            return out

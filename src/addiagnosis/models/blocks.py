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

from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv3d(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
) -> nn.Module:
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
    )


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_track_running_stats: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn_track_running_stats = bn_track_running_stats
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float = 0.05, stride: int = 1, 
                bn_track_running_stats: bool = True, no_downsample: bool = False):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.dropout1 = nn.Dropout(p=0.2, inplace=True)

        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.no_downsample = no_downsample

        if not no_downsample:
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
                )
            else:
                self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if not self.no_downsample:
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
        
        out = self.relu(out)

        return out

class Decoder_ResBlock(nn.Module):
    def __init__(self, in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 bn_momentum: float = 0.05, 
                 stride: int = 1, 
                 bn_track_running_stats: bool = True,
                ):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.conv1 = conv3d(in_channels + skip_channels, 
                            out_channels, stride=1)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.dropout1 = nn.Dropout(p=0.2, inplace=True)

        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.upsample_residual = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.upsample_residual = None
  
    
    def forward(self, x, skip = None):

        if skip is not None:
            target_size = skip.shape[-1]
            x = Upsample(x, target_size, skip.shape[1])
            residual = x 
            x = torch.cat([x, skip], dim=1)
        else:
            raise ValueError('no skip connected')
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample_residual is not None:
            residual = self.upsample_residual(residual)

        out += residual
        out = self.relu(out)

        return out

def Upsample(x, target_size, out_channels):
    in_channels = x.shape[1]
    upsample = nn.Upsample(target_size, mode = 'nearest')
    conv3d = nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = 1).to(device)
    bn = nn.BatchNorm3d(out_channels).to(device)
    relu = nn.ReLU().to(device)
    return relu(bn(conv3d(upsample(x)))).to(device)

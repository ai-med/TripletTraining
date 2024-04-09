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
from torch.nn.utils import weight_norm

class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [weight_norm(nn.Conv3d(in_channels = in_filters, out_channels = out_filters, 
                      kernel_size = 3, stride = stride, padding = 1))]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 2, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(weight_norm(nn.Conv3d(out_filters, 1, 3, 1, 1)))

        self.model = nn.Sequential(*layers)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
 
    def forward(self, img):
        out = self.global_pool(self.model(img))
        out = out.view(out.size(0), -1)
        return out
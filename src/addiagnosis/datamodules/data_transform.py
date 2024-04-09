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

import monai.transforms as montrans
import torchio as tio

class SSLTransform:
    def __init__(self, 
                 is_training = True, 
                 original_height = 120, 
                 input_height = 55, 
                 output_origin = False, 
                 roi_size = None
                ):

        self.original_height = original_height
        self.input_height = input_height
        self.is_training = is_training
        self.output_origin = output_origin
        self.roi_size = roi_size

        self.origin_transform = montrans.Compose([
             montrans.Resize((self.input_height, self.input_height, self.input_height)),
             montrans.ScaleIntensity(minv=0.0, maxv=1.0),
            ])

        self.crop_base_transform = montrans.Compose([
             montrans.RandSpatialCrop(roi_size = (roi_size, roi_size, roi_size),
                                      random_center = True, random_size = False),
             montrans.Resize((self.input_height, self.input_height, self.input_height)), 
             montrans.ScaleIntensity(minv=0.0, maxv=1.0),
            ])


        self.crop_final_transform = montrans.Compose(
            [              
                tio.RandomFlip(axes = (0,1,2), flip_probability=0.5),
                tio.RandomAffine(
                        scales=0.05,
                        degrees=90,  # +-90 degree in each dimension
                        translation=8,  # +-8 pixels offset in each dimension.
                        image_interpolation="linear",
                        default_pad_value="otsu",
                        p=0.5,
                        ),
            ]
        )

    def __call__(self, sample):
        crop_1_origin = self.crop_base_transform(sample)
        crop_1 = self.crop_final_transform(crop_1_origin)
        crop_2 = self.crop_final_transform(self.crop_base_transform(sample))
        if self.output_origin:
            return crop_1, crop_2, self.origin_transform(sample)
        else:
            return crop_1, crop_2, crop_1_origin



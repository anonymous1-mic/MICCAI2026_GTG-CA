from monai.transforms import (
     ToTensord,
    RandSpatialCropd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandRotate90d,
    RandCoarseDropoutd,
)
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, RandRotate90d, NormalizeIntensityd,
    RandSpatialCropd, CenterSpatialCropd, RandCoarseDropoutd, ToTensord, Lambda,Resized,RandScaleIntensityd,RandShiftIntensityd, RandFlipd, RandCropByPosNegLabeld,SpatialPadd
)
import numpy as np
import torch





class FullMaskGenerator:
    def __init__(self, patch_size=16, mask_ratio=0.4, device="cuda"):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.device = device

    def __call__(self, img_shape, batch_size):
        """
        Args:
            img_shape: (C, H, W, D)
            batch_size: B

        Returns:
            mask: (B, C, H, W, D)
        """
        C, H, W, D = img_shape
        ps = self.patch_size

        # number of patches
        Hp = (H + ps - 1) // ps
        Wp = (W + ps - 1) // ps
        Dp = (D + ps - 1) // ps
        total_patches = Hp * Wp * Dp
        num_masked = int(self.mask_ratio * total_patches)

        # ---- PATCH MASK (B, Hp, Wp, Dp)
        patch_mask = torch.zeros(
            (batch_size, total_patches),
            device=self.device
        )

        rand_idx = torch.rand(batch_size, total_patches, device=self.device).argsort(dim=1)
        patch_mask.scatter_(1, rand_idx[:, :num_masked], 1.0)
        patch_mask = patch_mask.view(batch_size, Hp, Wp, Dp)

        # ---- EXPAND TO VOXEL SPACE
        mask = patch_mask.repeat_interleave(ps, 1)\
                          .repeat_interleave(ps, 2)\
                          .repeat_interleave(ps, 3)

        # crop to original size
        mask = mask[:, :H, :W, :D]

        # expand channels
        mask = mask.unsqueeze(1).expand(-1, C, -1, -1, -1)

        return mask


import matplotlib.pyplot as plt

def visualize_mask_slice(mask_tensor, slice_idx=80, channel=0, axis='axial'):
    """
    Display a single slice of a 3D mask tensor.

    Args:
        mask_tensor (torch.Tensor): Shape [1, C, H, W, D]
        slice_idx (int): Slice index along chosen axis
        channel (int): Channel to display
        axis (str): 'axial', 'sagittal', or 'coronal'
    """
    mask_np = mask_tensor[0, channel].cpu().numpy()  # Shape: [H, W, D]

    if axis == 'axial':      # XY slice at Z = slice_idx
        slice_img = mask_np[:, :, slice_idx]
    elif axis == 'coronal':  # XZ slice at Y = slice_idx
        slice_img = mask_np[:, slice_idx, :]
    elif axis == 'sagittal': # YZ slice at X = slice_idx
        slice_img = mask_np[slice_idx, :, :]
    else:
        raise ValueError("Axis must be one of: 'axial', 'sagittal', 'coronal'.")

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_img, cmap='gray')
    plt.title(f"Mask Slice ({axis} axis), Slice {slice_idx}")
    plt.axis("off")
    plt.show()


    
from monai.transforms import MapTransform
import numpy as np

class ConvertToMultiChannelBasedOnCustomBratsClassesd(MapTransform):
    """
    Converts label values to multi-channel format for BraTS-like task.
    Your dataset label IDs:
    - 1: necrosis/NCR
    - 2: edema
    - 3: enhancing tumor (ET)

    Channels:
    - Channel 0: Tumor Core (TC) = 1 + 3
    - Channel 1: Whole Tumor (WT) = 1 + 2 + 3
    - Channel 2: Enhancing Tumor (ET) = 3
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]  # (C, H, W, D) or (H, W, D)
            
            if isinstance(seg, torch.Tensor):
                seg = seg.numpy()
            
           
            
            
            # make sure we're working with 3D (no extra channel dim)
            if seg.ndim == 4 and seg.shape[0] == 1:
                seg = np.squeeze(seg, axis=0)
            
            seg = np.where(seg == 4, 3, seg)
            tc = np.logical_or(seg == 1, seg == 3)   # Tumor Core
            wt = np.logical_or(tc, seg == 2)         # Whole Tumor
            et = seg == 3                             # Enhancing Tumor

            multi_channel = np.stack([tc, wt, et], axis=0).astype(np.float32)  # (3, H, W, D)
            d[key] = multi_channel
        return d
# For training (includes segmentation if available)
def print_shape(d):
    for k, v in d.items():
        print(f"{k}: {v.shape}")
    return d


from monai.transforms import MapTransform

# #Load biobert features
# class LoadNumpyd(MapTransform):
#     def __init__(self, keys):
#         super().__init__(keys)

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             d[key] = np.load(d[key])
#             d[key] = np.squeeze(d[key],axis=0)
#             d[key] = np.mean(d[key], axis=0)
#         return d


from monai.transforms import MapTransform

class LoadNumpyd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in data and allow_missing_keys=False")

            arr = np.load(d[key])  # (1, 128, 768)
            arr = np.squeeze(arr, axis=0)  # (128, 768)
            arr = arr.astype(np.float32)

            d[key] = arr
        return d



train_transforms = Compose([
    LoadImaged(keys=["img", "seg"], allow_missing_keys=True, ensure_channel_first=True),
    LoadNumpyd(keys=["text_feature"],allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg", allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    RandRotate90d(keys=["img", "seg"], prob=0.5, allow_missing_keys=True),
    RandSpatialCropd(keys=["img", "seg"], roi_size=(128,128,128), random_center=True, random_size=False, allow_missing_keys=True),
    ToTensord(keys=["img", "seg","text_feature"], dtype=torch.float32, allow_missing_keys=True),  # include is_dummy
])    



val_transforms = Compose([
    LoadImaged(keys=["img", "seg"], ensure_channel_first=True,allow_missing_keys=True),
    LoadNumpyd(keys=["text_feature"],allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg",allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img",  "seg","text_feature"],dtype=torch.float32, allow_missing_keys=True),
])




test_transforms = Compose([
    LoadImaged(keys=["img","seg"], ensure_channel_first=True,allow_missing_keys=True),
    LoadNumpyd(keys=["text_feature"],allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg",allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img","seg","text_feature"],dtype=torch.float32,allow_missing_keys=True),
])

































# Set random seed for reproducibility
# test_transforms.transforms[2].set_random_state(seed=48)  # RandCoarseDropoutd

# # Define transformations
# train_transforms = Compose([
#     LoadImaged(keys=["img", "groundtruth"], ensure_channel_first=True),
#     ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
#     RandRotate90d(keys=["img", "groundtruth"], prob=0.5),
#     RandSpatialCropd(keys=["img", "groundtruth"], roi_size=(128, 160, 128)),
#     RandCoarseDropoutd(keys=["img"], holes=500, spatial_size=(20, 20, 20), fill_value=0),
#     ToTensord(keys=["img", "groundtruth"]),
# ])


# val_transforms = Compose([
#     LoadImaged(keys=["img", "groundtruth"], ensure_channel_first=True),
#     ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
#     RandRotate90d(keys=["img", "groundtruth"], prob=0.5),
#     RandSpatialCropd(keys=["img", "groundtruth"], roi_size=(128, 160, 128)),
#     RandCoarseDropoutd(keys=["img"], holes=500, spatial_size=(20, 20, 20), fill_value=0),
#     ToTensord(keys=["img", "groundtruth"]),
# ])

# test_transforms = Compose([
#     LoadImaged(keys=["img", "groundtruth"], ensure_channel_first=True),  
#     ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),  # Rescale both image and groundtruth to [0, 1]
#     RandCoarseDropoutd(keys=["img"], holes=500, spatial_size=(20, 20, 20), fill_value=0),
#     ToTensord(keys=["img", "groundtruth"]),
# ])

import numpy as np

import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib

from monai.inferers import SlidingWindowInferer, sliding_window_inference
from torch.nn import CrossEntropyLoss

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import AsDiscrete, Compose, EnsureType
from collections import OrderedDict
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.metrics import compute_hausdorff_distance
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

# Define tensor-level TTA transforms as lambdas
tta_transforms = [
    lambda x: x,  # identity
    lambda x: torch.flip(x, dims=[2]),  # flip x-axis (assuming C,H,W,D format -> H=2)
    lambda x: torch.flip(x, dims=[3]),  # flip y-axis
    lambda x: torch.flip(x, dims=[4]),  # flip z-axis
    lambda x: torch.rot90(x, k=1, dims=[3,4]),  # rotate 90° around y-z plane
]

def invert_tta(x, t):
    # Inverse operations for each transform, match order with tta_transforms list

    # Identity inverse is identity
    if t == tta_transforms[0]:
        return x
    # flip is its own inverse
    elif t == tta_transforms[1]:
        return torch.flip(x, dims=[2])
    elif t == tta_transforms[2]:
        return torch.flip(x, dims=[3])
    elif t == tta_transforms[3]:
        return torch.flip(x, dims=[4])
    elif t == tta_transforms[4]:
        # inverse rotation is rotating 3 times (270 degrees)
        return torch.rot90(x, k=3, dims=[3,4])
    else:
        return x
def convert_to_single_channel(multi_channel_np: np.ndarray) -> np.ndarray:
    """
    Convert BraTS-style one-hot (3, H, W, D) prediction or GT to single-channel label map:
        0: Background
        1: Tumor Core (TC) [label 1 in GT]
        2: Edema [label 2 in GT]
        3: Enhancing Tumor (ET) [label 3 in GT]

    Assumes:
        Channel 0: TC = 1 + 3
        Channel 1: WT = 1 + 2 + 3
        Channel 2: ET = 3
    """
    assert multi_channel_np.shape[0] == 3, "Expected 3 channels (TC, WT, ET)"
    
    tc = multi_channel_np[0]
    et = multi_channel_np[2]

    output = np.zeros_like(tc, dtype=np.uint8)

    # Priority-based assignment
    output[tc == 1] = 1  # TC gets label 1 (includes necrosis and ET)
    output[(multi_channel_np[1] == 1) & (tc == 0) & (et == 0)] = 2  # Edema only gets label 2
    output[et == 1] = 3  # Enhancing Tumor gets label 3 (overwrites TC if needed)

    return output
def test(test_loader, model, input_dir, results_dir):
    import os
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.utils.enums import MetricReduction
    import time

    # --- AverageMeter class ---
    class AverageMeter(object):
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

    device = next(model.parameters()).device
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    # --- Metrics ---
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    hd95_metric = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, percentile=95.0)

    # --- AverageMeters ---
    run_dice = AverageMeter()
    run_hd95 = AverageMeter()
    run_hd95.sum = np.zeros(3, dtype=np.float64)
    run_hd95.count = np.zeros(3, dtype=np.float64)
    run_hd95.avg = np.zeros(3, dtype=np.float64)

    from functools import partial
    from monai.inferers import sliding_window_inference
    # --- sliding window predictor ---
   

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch in enumerate(test_loader):
            img = batch["img"].to(device)
            gt = batch["seg"].to(device)
            text = batch["text_feature"].to(device)
            
            
            predictor_with_text = lambda x: model(x, text)[0]

            model_inferer_with_text = partial(
                sliding_window_inference,
                roi_size=[128, 128, 128],
                sw_batch_size=2,
                predictor=predictor_with_text,
                overlap=0.7,
                # mode="gaussian"
            )

            # --- sliding window inference ---
            logits = model_inferer_with_text(img)
            
            # logits=model(img,text)
            pred_prob = torch.sigmoid(logits)
            pred = (pred_prob > 0.5).int()
    
            
            
            
            
            subject_id = batch["subject_id"][0]  # The first element in the batch
            print(subject_id,pred.shape)
            affine_modality = "0001"
            img_path = os.path.join(input_dir,  f"{subject_id}_{affine_modality}.nii.gz")
            import nibabel as nib
            affine = nib.load(img_path).affine

        
            save_filename = f"{subject_id}"
            save_filename_gt = f"{subject_id}_gt"
    
            import numpy as np
           
            
            save_pred_path = os.path.join(results_dir, f"{save_filename}.nii.gz")
            save_gt_path = os.path.join(results_dir, f"{save_filename_gt}.nii.gz")
            affine = nib.load(img_path).affine
            # Save the images
            # nib.save(nib.Nifti1Image(img_np, affine), save_img_path)
            # After converting predictions to numpy
            pred_np = pred.cpu().numpy().astype(np.uint8)
            seg_np = gt.cpu().numpy().astype(np.uint8)
            # print('check shape again:',pred_np.shape,seg_np.shape)
            
            # Convert to single-channel (with correct label encoding)
            single_channel_pred = convert_to_single_channel(pred_np[0])
            single_channel_gt = convert_to_single_channel(seg_np[0])  # If saving GT
            # print("Pred unique labels:", np.unique(single_channel_pred))
            # print("GT   unique labels:", np.unique(single_channel_gt))

            
            # # # Save NIfTI
            nib.save(nib.Nifti1Image(single_channel_pred, affine), save_pred_path)
            nib.save(nib.Nifti1Image(single_channel_gt, affine), save_gt_path)
            
  
        
            dice_metric(y_pred=pred, y=gt)
            dice, not_nans = dice_metric.aggregate()
            
            
            # print('check dice..............................:',dice)
            
            
            
            
            dice = dice.cpu().numpy()
            run_dice.update(dice, n=not_nans.cpu().numpy())

            # --- HD95 per batch (no reset) ---
            hd95_metric(y_pred=pred, y=gt)
            hd95 = hd95_metric.aggregate().detach().cpu().numpy()  # (3,)
           
            
           
            # print("case WT HD95:", hd95.shape, "case id:", subject_id)
            # print(f"[{subject_id}] HD95 case: TC={hd95[0]:.2f}, WT={hd95[1]:.2f}, ET={hd95[2]:.2f}")            

            valid_mask = np.isfinite(hd95)
            
            # update only valid classes
            run_hd95.sum[valid_mask] += hd95[valid_mask]
            run_hd95.count[valid_mask] += 1.0
            
            run_hd95.avg = np.where(run_hd95.count > 0, run_hd95.sum / run_hd95.count, 0.0)
            
        

            Dice_TC, Dice_WT, Dice_ET = run_dice.avg
            HD95_TC, HD95_WT, HD95_ET = run_hd95.avg
            print(
                f"Batch {batch_idx+1}/{len(test_loader)} | "
                f"Dice -> TC: {Dice_TC:.4f}, WT: {Dice_WT:.4f}, ET: {Dice_ET:.4f} | "
                f"HD95 -> TC: {HD95_TC:.2f}, WT: {HD95_WT:.2f}, ET: {HD95_ET:.2f} | "
                f"time: {time.time()-start_time:.2f}s"
            )
            start_time = time.time()
    
    # --- final print and return ---
    print(
        f"\nFinal Validation Results - Dice_TC: {Dice_TC:.4f}, Dice_WT: {Dice_WT:.4f}, Dice_ET: {Dice_ET:.4f}, "
        f"Avg Dice: {(Dice_TC+Dice_WT+Dice_ET)/3:.4f}, "
        f"HD95_TC: {HD95_TC:.2f}, HD95_WT: {HD95_WT:.2f}, HD95_ET: {HD95_ET:.2f}, "
        f"Avg HD95: {(HD95_TC+HD95_WT+HD95_ET)/3:.2f}"
    )
    
    return run_dice.avg, run_hd95.avg































    #         # --- print per batch ---
    #         print(
    #             f"Batch {batch_idx+1}/{len(test_loader)} | "
    #             f"Dice -> TC: {dice[0]:.4f}, WT: {dice[1]:.4f}, ET: {dice[2]:.4f} | "
    #             f"HD95 -> TC: {hd95[0]:.2f}, WT: {hd95[1]:.2f}, ET: {hd95[2]:.2f} | "
    #             f"time: {time.time()-start_time:.2f}s"
    #         )
    #         start_time = time.time()

    # # --- final results from AverageMeter ---
    # mean_dice = run_dice.avg
    # mean_hd95 = run_hd95.avg

    # print("\nFinal Dataset Avg Dice -> "
    #       f"TC={mean_dice[0]:.4f}, WT={mean_dice[1]:.4f}, ET={mean_dice[2]:.4f}")
    # print("Final Dataset Avg HD95 -> "
    #       f"TC={mean_hd95[0]:.2f}, WT={mean_hd95[1]:.2f}, ET={mean_hd95[2]:.2f}")

    # # --- log to file ---
    # with open(os.path.join(results_dir, 'log.txt'), "a") as f:
    #     f.write(
    #         f"Final Validation Results | "
    #         f"Dice_TC: {mean_dice[0]:.4f}, Dice_WT: {mean_dice[1]:.4f}, Dice_ET: {mean_dice[2]:.4f}, "
    #         f"Avg Dice: {mean_dice.mean():.4f} | "
    #         f"HD95_TC: {mean_hd95[0]:.2f}, HD95_WT: {mean_hd95[1]:.2f}, HD95_ET: {mean_hd95[2]:.2f}, "
    #         f"Avg HD95: {mean_hd95.mean():.2f}\n"
    #     )

    # return mean_dice, mean_hd95

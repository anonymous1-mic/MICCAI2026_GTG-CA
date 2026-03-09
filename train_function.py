import monai
import torch
# import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
from monai.metrics import SSIMMetric
from load_data_multitask1 import *
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from transforms_multitask import *
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from functools import partial
from monai.inferers import sliding_window_inference
from functools import partial
import torch.nn.functional as F
import torch.distributed as dist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from monai.losses import DiceLoss


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

def clip_contrastive_loss(image_emb, text_emb, temperature=0.07):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb  = F.normalize(text_emb,  dim=-1)
    # print('check shape_image_text,',image_emb,text_emb.shape)

    # similarity matrix [B, B]
    logits = torch.matmul(image_emb, text_emb.T) / temperature
    # print('check shape_logit_label,',logits.shape,logits)
    
    labels = torch.arange(logits.size(0), device=logits.device)
    # print('check shape_labelsval,',labels.shape,labels)

    loss_i2t = F.cross_entropy(logits, labels)
    # print('check shape_l1,',loss_i2t)
    loss_t2i = F.cross_entropy(logits.T, labels)
    # print('check shape_l2,',loss_t2i)
    return 0.5 * (loss_i2t + loss_t2i)


#############with prior#############
def empirical_corr(z):
    """
    z: (B, D) text embeddings
    """
    # subtract per-sample mean across feature dimension
    z_mean = z.mean(dim=1, keepdim=True)
    z_centered = z - z_mean

    # normalize
    z_norm = z_centered / (z_centered.norm(dim=1, keepdim=True) + 1e-8)

    # correlation matrix
    R = z_norm @ z_norm.T   # (B, B)
    return R

def smooth_corr(R, lambda_=0.2):
    B = R.size(0)
    R_smooth = 1.0 - torch.exp(-lambda_ * R)
    R_smooth.fill_diagonal_(1.0)
    return R_smooth


def smooth_corr(R, tmin=0.2, tmax=1.0):
    """
    R: (B, B) cosine-like similarity matrix in [-1, 1]
    returns: strictly positive matrix usable as dynamic temperature
    """
    # map from [-1, 1] → [0, 1]
    S = (R + 1.0) * 0.5

    # clamp for numerical safety
    S = torch.clamp(S, min=1e-4, max=1.0)

    # map to temperature range
    T = tmin + (tmax - tmin) * S

    # strengthen positive pairs
    T.fill_diagonal_(tmin)

    return T




def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, output_dir, start_epoch=1,rank=0):
    import torch.nn as nn
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 
   # model = model.to(device)            # Move model to cuda:0
    model = nn.DataParallel(model)

 
    results_dir=output_dir
    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    class_weights = torch.tensor([0.2, 0.1, 0.7], device=device).view(1, 3, 1, 1, 1)
    criterion_ce = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    last_model_path = os.path.join(output_dir, "last_model.pth")
    training_results_dir = os.path.join(output_dir, "training")
    os.makedirs(directory_name, exist_ok=True)
    os.makedirs(training_results_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_dice_score=-1.0
    check=last_model_path
    if os.path.exists(check):
        checkpoint = torch.load(check, map_location=device)
        state_dict = checkpoint['state_dict']
        
        # ########just to avoid DataParallel during re-training 
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k.replace('module.', '')  # remove 'module.' if exists
        #     new_state_dict[name] = v
        
        # # Load the state dict
        # model.load_state_dict(new_state_dict)
        
        ###########################################3
     
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        best_dice_score=checkpoint.get('best_dice_score',-1)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Last model loaded. Resuming training from epoch: {start_epoch}")
        print(f"Resuming with best Dice score: {best_dice_score:.4f}")




   
    def compute_binary_dice(pred, gt, epsilon=1e-5):
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)
        intersection = (pred & gt).sum()
        return (2.0 * intersection) / (pred.sum() + gt.sum() + epsilon)
    
    def get_alpha_beta(epoch, seg_start=50, max_epoch=200):
        if epoch < seg_start:
            return 0.0, 1.0
        progress = min((epoch - seg_start) / (max_epoch - seg_start), 1.0)
        beta = 0.3 * (1.0 - progress)  # recon decays to 0
        alpha = 1.0 - beta             # seg increases to 1
        return alpha, beta
    
    
    def get_gamma(epoch, clip_start=75, clip_max=0.2, max_epoch=200):
        if epoch < clip_start:
            return 0.0
        progress = min((epoch - clip_start) / (max_epoch - clip_start), 1.0)
        return clip_max * (progress ** 2.0)  # linear

    


    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\n🔁 Epoch {epoch}")
        model.train()
        train_loss = 0.0
    

        for batch in train_loader:
            img = batch["img"].to(device)
            seg = batch.get("seg", None).to(device)
            text=batch.get("text_feature", None).to(device)
            # has_text = batch["has_text"].to(device)
            
            # print('check image size',img.shape,seg.shape,text.shape)

           
            # print("Min/max/mean before network:", img.min(), img.max(), img.mean())

            groundtruth = img

            if seg is not None:
                seg = seg.to(device)
            
            if text is not None:
                text = text.to(device)
            B, C, H, W, D = img.shape
            
        
            # -------------------------
            if epoch < 50:
                mask_ratio = 0.40
            elif epoch < 100:
                mask_ratio = 0.25
            elif epoch < 150:
                mask_ratio = 0.10
            elif epoch < 200:
                mask_ratio = 0.05
            else:
                mask_ratio = 0.0
            
            
            # -------------------------
            # Generate mask
            # -------------------------
            if mask_ratio > 0.0:
                mask_generator = FullMaskGenerator(
                    patch_size=16,
                    mask_ratio=mask_ratio,
                    device=img.device
                )
            
                with torch.no_grad():
                    mask = mask_generator((C, H, W, D), batch_size=B)  # (B, C, H, W, D)
            else:
                mask = torch.zeros_like(img)
            
            
            # -------------------------
            # Apply mask
            # -------------------------
            masked_input = img * (1.0 - mask)
            
            optimizer.zero_grad()
            
                    
                    
            
            pred_seg, pred_recon, text_emb, image_emb= model(masked_input,text)
            # print('text_emb size check:',text_emb,image_emb,pred_seg.mean().item(),pred_recon.mean().item())

            if epoch < 50:
               
                loss = F.l1_loss(pred_recon, img)
                train_loss += loss.item()
            else:
                alpha,beta=get_alpha_beta(epoch, seg_start=50, max_epoch=200)
                loss = 0.0
                loss_seg = torch.tensor(0.0, device=img.device)
        
                loss_recon = F.l1_loss(pred_recon, img)
                loss += beta * loss_recon 
                # print('check loss_recon:',loss_recon)
                # 
                valid_mask = ~batch["is_dummy"].bool().to(device)
                print('check valid mask:', valid_mask)
                if valid_mask.any():
                    # Compute loss for all valid samples at once
                    loss_seg = criterion(pred_seg[valid_mask], seg[valid_mask])
                    loss_seg += criterion_ce(pred_seg[valid_mask], seg[valid_mask])
                    loss += alpha * loss_seg 
                    
                
                valid_indices = (~batch["is_dummy"].bool()).nonzero(as_tuple=True)[0]
                # print('check if valid:',valid_indices)

                if len(valid_indices) >=2:
                    print('check if valid:',valid_indices)
                    image_valid = image_emb[valid_indices]
                    text_valid = text_emb[valid_indices]
                    R = empirical_corr(text_valid)
                    R_smooth = smooth_corr(R).detach()

                    loss_sim = clip_contrastive_loss(image_valid, text_valid, R_smooth)
                    # print('check if clip is used:',loss_sim)
                    gamma = get_gamma(epoch, clip_start=75, clip_max=0.2, max_epoch=200)
                    # print('check if clip is used:',gamma*loss_sim)
                    loss += gamma * loss_sim

                
                train_loss += loss.item()
            loss.backward()
           
            optimizer.step()
           

            

        train_loss /= len(train_loader)
        print(f"✅ Training Loss: {train_loss:.4f}")

 
        # ----------------------
        # Validation
        # ----------------------
        model.eval()

        affine = np.eye(4)
        
        with torch.no_grad():
            dice_metric.reset()
            for batch_idx, batch in enumerate(val_loader):
               
                img = batch["img"].to(device)
                seg = batch.get("seg", None)
                groundtruth = img    
                if seg is not None:
                    seg = seg.to(device)

                B, C, H, W, D = img.shape
                #print("Min/max/mean before network during val:", img.min(), img.max(), img.mean(),img.shape)

                
                text = batch.get("text_feature", None)
                
                if text is not None:
                    text = text.to(device)
             
                predictor_with_text = lambda x: model(x, text)[0]
                
                # Create a sliding_window_inference instance with this predictor
                model_inferer_with_text = partial(
                    sliding_window_inference,
                    roi_size=[128,128,128],
                    sw_batch_size=2,
                    predictor=predictor_with_text,
                    overlap=0.7,
                )
                
                # Run inference
                pred_seg = model_inferer_with_text(img)
                # if pred_seg.sum() != 0:
                #     print("Seg is not zero!", pred_seg.shape, text.shape, seg.shape)
                    
                # for i, p in enumerate(pred_seg):
                #     print(f"Sample {i}: min={p.min().item():.4f}, max={p.max().item():.4f}, mean={p.mean().item():.4f}")

                
             
                if seg is not None:
                    # print('entered')
                    is_dummy = batch["is_dummy"].to(device)
                    valid_mask = ~is_dummy
                    #print(valid_mask,is_dummy)
                    if valid_mask.sum() > 0:
                        # print('true...................................................')
                        # Prepare lists of valid predictions and labels
                        val_output_convert = [post_pred(post_sigmoid(p)) for p in pred_seg]
                        pred_seg = [p for p, d in zip(val_output_convert, is_dummy) if not d]
                        gt_seg = [s for s, d in zip(seg, is_dummy) if not d]
                        
                     
                        # for p, g in zip(pred_seg, gt_seg):
                        #     print("pred sum:", p.sum().item(), "gt sum:", g.sum().item(),
                        #           "overlap:", (p * g).sum().item())

                        # Accumulate metric
                        dice_metric(y_pred=pred_seg, y=gt_seg)
    
                        # Save NIfTI for valid samples
                        for idx, i in enumerate(valid_mask.nonzero(as_tuple=True)[0]):
                            subject_id = batch["subject_id"][i]
                            img_path = os.path.join(directory_name, f"{subject_id}_0000.nii.gz")
                            save_pred_path = os.path.join(results_dir, f"{subject_id}_pred.nii.gz")
                            save_img_path = os.path.join(results_dir, f"{subject_id}_gt.nii.gz")
    
                            affine = nib.load(img_path).affine
                            pred_np = pred_seg[idx].detach().cpu().numpy().astype(np.uint8)
                            seg_np = gt_seg[idx].detach().cpu().numpy().astype(np.uint8)
    
                            single_channel_pred = convert_to_single_channel(pred_np)
                            single_channel_gt = convert_to_single_channel(seg_np)
    
                            nib.save(nib.Nifti1Image(single_channel_pred, affine), save_pred_path)
                            nib.save(nib.Nifti1Image(single_channel_gt, affine), save_img_path)
    
        # Aggregate Dice once for the epoch
        if seg is not None and valid_mask.sum() > 0:
            print('entered...................')
            per_class_dice, _ = dice_metric.aggregate()
            mean_dice = per_class_dice.mean().item()
            print(f"📊 Dice Scores — TC: {per_class_dice[0].item():.4f}, "
                  f"WT: {per_class_dice[1].item():.4f}, ET: {per_class_dice[2].item():.4f}")
            print(f"🌟 Mean Dice: {mean_dice:.4f}")
        else:
            mean_dice = 0.0
    
        # Save best model
        if mean_dice > best_dice_score:
            best_dice_score = mean_dice
            torch.save({
                'epoch': epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice_score": best_dice_score
            }, checkpoint_path)
            print("✅ Best model saved based on Dice score.")
        
         
        # Always save last model
        torch.save({
            'epoch': epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice_score": best_dice_score
        }, last_model_path)
    
    
        scheduler.step()

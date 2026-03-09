
"""

"""

import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from load_data import *
from train_function import *
from textswin_unetr import *
from load_data_validation import load_data_validation  
from inference import *

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = TextSwinUNETR(
        img_size=(256, 256, 160),
        in_channels=4,
        out_channels=3,
        seg_out_channels=3,
        recon_out_channels=4,
        feature_size=48,
        text_dim=768,
        use_checkpoint=False,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)

    # =====================
    # RESUME OR TEST: LOAD CHECKPOINT
    # =====================
    if args.mode in ["resume", "test"]:
        assert args.checkpoint_dir is not None, "You must provide --checkpoint_dir for resume or test"
        model = model.to(device)            # Move model to cuda:0
        model = nn.DataParallel(model)
        checkpoint = torch.load(args.checkpoint_dir, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        best_dice_score = checkpoint.get("best_dice_score", -1)
        args.start_epoch = checkpoint.get("epoch", 0) + 1

        print(f"Loaded checkpoint from {args.checkpoint_dir}")
        print(f"Best Dice: {best_dice_score}")
        print(f"Resuming from epoch: {args.start_epoch}")

        if args.mode == "resume" and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

    # =====================
    # TRAIN
    # =====================
    if args.mode in ["train", "resume"]:
        train_loader, val_loader = load_data(
            args.image_dir, args.label_dir, args.text_dir, args.output_dir
        )

        train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=args.epochs,
            directory_name=args.image_dir,
            output_dir=args.output_dir,
            start_epoch=args.start_epoch,
        )

    # =====================
    # TEST
    # =====================
    if args.mode == "test":
        val_loader = load_data_validation(args.image_dir, args.label_dir, args.text_dir)

        model.eval()
        with torch.no_grad():
            test(val_loader, model, args.image_dir, args.output_dir)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTG-CA on TextBraTS dataset")
    
    # Data paths
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the dataset images directory (imagesTr)")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Path to the dataset label directory (labelsTr)")
    
    parser.add_argument("--text_dir", type=str, required=True,
                        help="Path to the text dataset directory (text_data)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs and models")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=250, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--t_max", type=int, default=50, help="T_max for CosineAnnealingLR")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum LR for scheduler")
    parser.add_argument("--start_epoch", type=int, default=1, help="Starting epoch")
    parser.add_argument("--checkpoint_dir",type=str,default=None,help="Path to checkpoint directory (only needed for testing or resuming)")
    parser.add_argument("--mode",type=str,default="train",choices=["train", "test", "resume"],
    help="Run mode: train from scratch, test only, or resume training"
)



    ##############running from terminal#################################
    args = parser.parse_args()
    main(args)


    #############running from Spyder####################################
    
    # import sys

    # if 'spyder' in sys.modules:
    #     class Args:
    #         image_dir = r"/path_to_image_dir"
    #         label_dir = r"/path_to_label_dir"
    #         text_dir  = r"/path_to_text_dir"
    #         output_dir = r"/path_to_result_dir"
    
    #         epochs = 250
    #         lr = 4e-4
    #         weight_decay = 1e-5
    #         t_max = 50
    #         eta_min = 1e-6
    #         start_epoch = 1
    #         checkpoint_dir = None
    #         mode = "train"   # change to "test" or "resume" if needed
    
    #     args = Args()
    # else:
    #     args = parser.parse_args()
    
    # main(args)

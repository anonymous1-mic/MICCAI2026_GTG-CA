import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import re
from monai.data import Dataset
from transforms_multitask import train_transforms, val_transforms, test_transforms
from torch.utils.data import DataLoader
def load_data(images_dir,
    labels_dir,
    text_features_dir,
    output_dir):
    """
    Load and split the dataset into training, validation, and test sets, save splits as CSV files,
    and configure DataLoaders for each. If test CSV exists, loads from it directly.
    """
    import nibabel as nib
    import numpy as np

    dummy_seg = np.zeros((256,256,160), dtype=np.uint8) 
    dummy_seg_path=os.path.join(output_dir, 'empty_seg.nii.gz')
    nii = nib.Nifti1Image(dummy_seg, affine=np.eye(4))
    print('check the dummy path...............:',dummy_seg_path)
    nib.save(nii, dummy_seg_path)
   
    
    
    import numpy as np
    dummy_text_path = os.path.join(output_dir, 'dummy_text.npy')
    np.save(dummy_text_path, np.zeros((1, 128, 768))) 

    
    path = images_dir
    label_path=labels_dir
 
    
    text_feature_root=text_features_dir

    val_csv_path = os.path.join(path, "validation_set.csv")
    train_csv_path = os.path.join(path, "train_set.csv")
    
    print('check path:',val_csv_path,train_csv_path)

    if os.path.exists(val_csv_path) and os.path.exists(train_csv_path):
        print("Loading existing splits...")
        
        val_data = pd.read_csv(val_csv_path)
        train_data = pd.read_csv(train_csv_path)
    else:
        print("No existing CSVs found. Performing new split...")
        all_files = sorted([f for f in os.listdir(path) if f.endswith('.nii.gz')])
        subject_ids = sorted(set(re.sub(r'_\d{4}\.nii\.gz$', '', f) for f in all_files))
        df = pd.DataFrame({'SubjectID': subject_ids})

        train_val_df, test_data = train_test_split(df, test_size=0.20, random_state=42)
        train_data, val_data = train_test_split(train_val_df, test_size=0.20, random_state=42)

        train_data['split'] = 'train'
        val_data['split'] = 'val'
        test_data['split'] = 'test'

        final_df = pd.concat([train_data, val_data, test_data]).sort_values(by='SubjectID')
        output_csv = os.path.join(output_dir, 'dataset_split.csv')
        final_df.to_csv(output_csv, index=False)

        # print(f"Split CSV saved to: {output_csv}")
        # print(final_df['split'].value_counts())

        # Save each split separately
        train_data.to_csv(train_csv_path, index=False)
        val_data.to_csv(val_csv_path, index=False)
        

    # Build final dataset structure
    def get_modality_paths(subject_id):
        return [
            os.path.join(path, f"{subject_id}_0000.nii.gz"),
            os.path.join(path, f"{subject_id}_0001.nii.gz"),
            os.path.join(path, f"{subject_id}_0002.nii.gz"),
            os.path.join(path, f"{subject_id}_0003.nii.gz")
        ]

    
   
    def build_data_list(df_split):
        data_list = []
        for sid in df_split['SubjectID']:
            img_paths = get_modality_paths(sid)
    
            seg_path = os.path.join(label_path, f"{sid}.nii.gz")
            seg_exists = os.path.exists(seg_path)
            
            text_feature_file = os.path.join(text_feature_root, sid, f"{sid}_flair_text.npy")
            text_exists = os.path.exists(text_feature_file)
    
            item = {
                "img": img_paths,
                "subject_id": sid,
                "seg": seg_path if seg_exists else dummy_seg_path,
                "text_feature": text_feature_file if text_exists else dummy_text_path,
                "is_dummy": not seg_exists,
                "has_text": text_exists
                
            }
    
            data_list.append(item)
        return data_list
    
    
    
    filenames_train = build_data_list(train_data)
    filenames_val = build_data_list(val_data)


    ds_train = Dataset(data=filenames_train, transform=train_transforms)
    ds_val = Dataset(data=filenames_val, transform=val_transforms)
    

    train_loader = DataLoader(ds_train, num_workers=4, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, num_workers=2, batch_size=2, shuffle=False,  drop_last=False)
    


    return train_loader, val_loader




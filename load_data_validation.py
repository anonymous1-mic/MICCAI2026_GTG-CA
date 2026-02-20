import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import re
def load_data_validation(image_dir,label_dir,text_dir):
    
    
    #############if following the same structure as training##########
    path = image_dir
    label_path=label_dir
    
    
    text_feature_root=text_dir

    test_csv_path = os.path.join(path, "test_set.csv")
   
    
    print('check path:',test_csv_path)

    if os.path.exists(test_csv_path):
        print("Loading existing splits...")
        
        test_data = pd.read_csv(test_csv_path)
      
    else:
        print("No existing CSVs found. Performing new split...")
        
  

        # Build final dataset structure
    def get_modality_paths(subject_id):
            return [
                os.path.join(path, f"{subject_id}_0000.nii.gz"),
                os.path.join(path, f"{subject_id}_0001.nii.gz"),
                os.path.join(path, f"{subject_id}_0002.nii.gz"),
                os.path.join(path, f"{subject_id}_0003.nii.gz")
            ]

        
       
    def build_data_list(test_data):
        data_list = []
        for sid in test_data['SubjectID']:
                img_paths = get_modality_paths(sid)
         
                seg_path = os.path.join(label_path, f"{sid}.nii.gz")
            
                
                text_feature_file = os.path.join(text_feature_root, sid, f"{sid}_flair_text.npy")
           
        
                item = {
                    "img": img_paths,
                    "subject_id": sid,
                    "seg": seg_path,
                    "text_feature": text_feature_file,
                }
         
                data_list.append(item)
        
        return data_list
         
         
         
    filenames_test = build_data_list(test_data)
     
    from transforms_multitask import test_transforms
    from monai.data import Dataset
    from torch.utils.data import DataLoader

    ds_test = Dataset(data=filenames_test, transform=test_transforms)

   
    test_loader = DataLoader(ds_test, num_workers=1, batch_size=1, shuffle=False,  pin_memory=True,drop_last=False)

    
    return test_loader
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from pathlib import Path
from .. import input_output as io

class CustomImageDataset(Dataset):
    def __init__(self, folder_input, folder_target):
        valid_suffix = {".tif", ".tiff"}
        
        self.folder_input = folder_input
        self.folder_target = folder_target
        self.data_augmentation_flag = False
        
        #variable to list all the files in the training set
        self.file_names = [p.name for p in Path(self.folder_input).iterdir() if p.suffix in valid_suffix]
        
        check_trainingset_file_matching(self.file_names, self.folder_target)        
    
    def __len__(self):
        return len(self.file_names)
    
    
    def __getitem__(self, index):
        
        input_img    = io.imread(Path(self.folder_input, self.file_names[index]))
        target_img = io.imread(Path(self.folder_target, self.file_names[index]))
        
        #converting numpy to tensor
        input_img = tensor(input_img.astype(np.float32)).float()
        target_img = tensor(target_img.astype(np.uint8))        
        
        #img is a 3D image with [D,W,H], we only have one channel image, change the dimensions to [1,D,W,H]
        input_img = input_img.unsqueeze(0) if input_img.dim() ==  3 else input_img
        target_img = target_img.unsqueeze(0) if target_img.dim() == 3 else target_img
        
        if self.data_augmentation_flag:
            input_img, target_img = self.data_augmentation_object.run(input_img, target_img)        
        
        return input_img, target_img      
    
    def set_data_augmentation(self, augmentation_flag = False, data_augmentation_object = None):
        """
        this method is used to set a data augmentation flag and object. 
        The data_augmentation_flag is a boolean indicating whether data augmentation should be performed or not
        data_augmentation_object is an object containing the data augmentation methods to be applied.
        """
        
        self.data_augmentation_flag = augmentation_flag
        self.data_augmentation_object = data_augmentation_object    
        
        
def check_trainingset_file_matching(file_names, folder_target):
    #verify each image in input_folder has an associated image in the target folder
    missing_files = [f for f in file_names if not Path(folder_target, f).is_file()]
    if missing_files:
        raise ValueError('Missing traces for images: ' + ', '.join(missing_files))
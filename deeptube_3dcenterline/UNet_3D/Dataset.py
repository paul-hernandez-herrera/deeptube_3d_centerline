import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from pathlib import Path
from .. import input_output as io

class CustomImageDataset3D(Dataset):
    def __init__(self, folder_input, folder_target):
        valid_suffix = [".tif", ".tiff"]
        
        #global variables
        self.folder_input = folder_input
        self.folder_target = folder_target
        
        #variable to list all the files in the training set
        self.file_names = [p.name for p in Path(self.folder_input).iterdir() if p.suffix in valid_suffix]
        
        #verify each image in input_folder has an associated image in the target folder
        verify_files_in_target_folder(self.file_names, self.folder_target)        
    
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
        target_img = target_img.unsqueeze(0) if target_img.dim() == 2 else target_img
        
        return input_img, target_img
        
        
        
        
        
def verify_files_in_target_folder(file_names, folder_target):
    #verify each image in input_folder has an associated image in the target folder
    for f in file_names:
        current_file = Path(folder_target, f)
        if not(current_file.is_file()):
            raise ValueError('Missing trace for image: ' + f + '\nRequired file: ' +  str(current_file) + '\n' )
from torch.utils.data import Dataset
from pathlib import Path
from . import input_output as io

class trainingset_dataset_3D(Dataset):
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
        input_target = io.imread(Path(self.folder_target, self.file_names[index]))
        
        return input_img, input_target
        
        
        
        
        
def verify_files_in_target_folder(file_names, folder_target):
    #verify each image in input_folder has an associated image in the target folder
    for f in file_names:
        current_file = Path(folder_target, f)
        if not(current_file.is_file()):
            raise ValueError('Missing trace for image: ' + f + '\nRequired file: ' +  str(current_file) + '\n' )
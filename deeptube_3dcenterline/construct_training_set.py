import numpy as np
from pathlib import Path
from .util import util as io
import warnings
from .util.preprocess import preprocess_image

#from . import util

class construct_training_set():
    
    def __init__(self):
        self.set_default_values()

        
    def set_default_values(self):
        #Default values for the class
        self.folder_traces = ''
        self.folder_imgs = ''
        self.folder_output = ''
        self.patch_size_img = 128
        self.number_patches = 10
        self.number_patches_random_pos = 2
        self.radius_tubular_mask = 4
        self.norm_perc_low = 1
        self.norm_perc_high = 99
        self.draw_head = True
        self.file_names = []
        
        return 

    def set_number_patches_random_pos(self, val):
        self.number_patches_random_pos = val
        return        

    def set_folder_traces(self, val):
        self.folder_traces = Path(val)
        return

    def set_folder_imgs(self, val):
        self.folder_imgs = Path(val)
        return
    
    def set_folder_output(self,val):
        self.folder_output = Path(val)
        return    
        
    def set_patch_size_img(self, val):
        self.patch_size_img = val
        return
        
    def set_number_patches(self, val):
        self.number_patches = val
        return
        
    def set_radius_tubular_mask(self, val):
        self.radius_tubular_mask = val
        return

    def set_normalization_percentile_low(self, val):
        self.norm_perc_low = val
        return

    def set_normalization_percentile_high(self, val):
        self.norm_perc_high = val
        return

    def set_draw_head(self, val):
        self.draw_head = val
        return

    def percentile_normalization(self, img, p_low, p_high):
        #normalize data to percentile and [0,1]
        low = np.percentile(img, p_low)
        high = np.percentile(img, p_high)
        
        normalize_img = (np.clip(img, low, high) - low)/(high-low)
        return normalize_img

    def run_main(self):
        self.core()
        
        
        
    ## PRIVATE METHODS
    def core(self):
        
        if not self.folder_output:
            self.folder_output = Path(self.folder_imgs, 'training_set')
            
        #get all file names in directoty
        self.file_names = [p.stem for p in Path(self.folder_imgs).glob('*.tif')]
        
        if not self.file_names:
            warnings.warn(f'folder images does not contain tif files\n{self.folder_imgs}')
        else:
            print(f'# images detected: {len(self.file_names)}\n')
            self.__verify_correct_traninig_set()            
        
        #make output folders
        folder_out_imgs = Path(self.folder_output, 'input')
        folder_out_masks = Path(self.folder_output, 'target')
        [self.__create_folder_if_not_exist(path) for path in (self.folder_output, folder_out_imgs, folder_out_masks)]

        count_img = 0
        n_files = len(self.file_names)
        
        for index, f_n in enumerate(self.file_names):
            print(f'Running stack: {index + 1}/{n_files}')
            img, swc = self.__get_data(f_n)
            
            #preprocess_data
            img = preprocess_image(img, percentile_range = [self.norm_perc_low, self.norm_perc_high], normalization_range = [0,1] )
            img = np.squeeze(img)
            
            #generate the ground true as a tubular structure
            mask = self.construct_tubular_mask(np.array(img.shape), swc, self.radius_tubular_mask)
            
            if self.draw_head:
                swc_head = swc[0:1,:]
                mask = mask | self.construct_tubular_mask(np.array(img.shape), swc_head , int(2.5*self.radius_tubular_mask))
            
            #generate training set images containing the flagellum
            for i in np.random.choice(swc.shape[0], self.number_patches, replace=False):
                left_upper_corner = swc[i,3:1:-1] - self.patch_size_img/2
                #random shift of the left-corner
                left_upper_corner += np.random.randint(low = -self.patch_size_img/4, high=self.patch_size_img/4, size=2, dtype=int)
                img_cropped, mask_cropped = self.__crop_subvolumes(img, mask, left_upper_corner, self.patch_size_img)
                io.imwrite(Path(folder_out_imgs, f"img_{count_img:06}.tif"), img_cropped)
                io.imwrite(Path(folder_out_masks, f"img_{count_img:06}.tif"), mask_cropped)
                count_img+=1

            #generate training set images containing the flagellum
            for i in range(0, self.number_patches_random_pos):
                left_upper_corner = np.array([np.random.choice(img.shape[2]- self.patch_size_img,1)[0], np.random.choice(img.shape[1]- self.patch_size_img,1)[0]])
                img_cropped, mask_cropped = self.__crop_subvolumes(img, mask, left_upper_corner, self.patch_size_img)
                io.imwrite(Path(folder_out_imgs, f"img_{count_img:06}.tif"), img_cropped)
                io.imwrite(Path(folder_out_masks, f"img_{count_img:06}.tif"), mask_cropped)
                count_img+=1                
                
        print('------------------------------')
        print('\033[47m' '\033[1m' 'Algorithm has finished generating training set.' '\033[0m')
        print('------------------------------')  
        print('\nTraining set saved in path: ')
        print(self.folder_output)
        print('\n')

        
    def __create_folder_if_not_exist(self, folder_path):
        folder_path.mkdir(parents=True, exist_ok=True)
        
    def __crop_subvolumes(self, img, mask, left_upper_corner, v_size):    
        #check good boundary conditions
        left_upper_corner = np.int_(left_upper_corner)
        v_size = np.int_(v_size)
        
        left_upper_corner[left_upper_corner<0] = 0
        
        #check the subvolume does not fall outside the img.shape
        img_2d_shape = np.array(img.shape[1:])
        outside_index = (left_upper_corner+ v_size)>img_2d_shape
        left_upper_corner[outside_index] = img_2d_shape[outside_index] - v_size
        
                
        
        img_cropped = img[:,left_upper_corner[0]:left_upper_corner[0]+ v_size,left_upper_corner[1]:left_upper_corner[1]+v_size]
        mask_cropped = mask[:,left_upper_corner[0]:left_upper_corner[0]+ v_size,left_upper_corner[1]:left_upper_corner[1]+v_size]
        
        return img_cropped, mask_cropped    
 
    
    def __get_data(self, file_name):
        img = io.imread(Path(self.folder_imgs, file_name + '.tif'))
        swc = io.read_swc(self.folder_traces , file_name + '.swc')
        return img, swc
    
    def construct_tubular_mask(self, img_shape, swc, r):
        #create a larger volume to have good boundary conditions
        cylinder_mask = np.full(img_shape+2*r, False)
        
        sphere = self.__construct_sphere(r)
        
        for i in range(0, swc.shape[0]):
            #updating coordinates
            x, y, z = swc[i,2:5]
            self.__merge_volumes(cylinder_mask, sphere, np.uint16(np.array([z,y,x])))
            
        return np.uint8(cylinder_mask[r:r+img_shape[0],r:r+img_shape[1],r:r+img_shape[2]]>0)
    
    def __construct_sphere(self, r):
        c = np.arange(-r,r+1)
        
        Z, X, Y = np.meshgrid(c,c,c)
        
        dist_ = np.sqrt(X**2 + Y**2 + Z**2)
        
        return dist_<=r
    
    def __merge_volumes(self, large_vol, small_vol, ini_pos):
        #We assume that small_volume is inside large volume (It is satisfied by construction)
        large_vol[ini_pos[0]:ini_pos[0]+small_vol.shape[0], ini_pos[1]:ini_pos[1]+small_vol.shape[1], ini_pos[2]:ini_pos[2]+small_vol.shape[2]] =  large_vol[ini_pos[0]:ini_pos[0]+small_vol.shape[0], ini_pos[1]:ini_pos[1]+small_vol.shape[1], ini_pos[2]:ini_pos[2]+small_vol.shape[2]] | small_vol
        return large_vol
    
    def __verify_correct_traninig_set(self):
        #verify each image has the corresponding swc trace
        for f_n in self.file_names:
            current_file_name = Path(self.folder_traces,f_n+'.swc')
            if not(current_file_name.is_file()):
                raise ValueError('Missing trace for image: ' + f_n + '.tif\nRequired file: ' +  str(current_file_name) + '\n' )
        return
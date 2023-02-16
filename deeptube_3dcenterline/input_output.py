import tifffile
import numpy as np
from pathlib import Path
import pandas as pd

def imread(filename):
    ext = Path(filename).suffix
    if ext== '.tif' or ext=='.tiff':
        return tifffile.imread(filename)
        
    
def imwrite(filename, arr):
    ext = Path(filename).suffix
    if ext== '.tif' or ext=='.tiff':
        tifffile.imsave(filename, arr) 
        
        
def read_swc(file_path,file_name):
    return np.array(pd.read_csv(Path(file_path) / file_name, header = None, comment='#', delim_whitespace = True))
        
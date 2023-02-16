import torch.nn as nn

class UNet_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #this block allows to initialize the parameters of the network
        
        
class Double_convolution(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2):
        super().__init__()
        
        self.double_conv3D = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = num_filters_conv1, kernel_size = (3,3,3), padding='same'),
            nn.BatchNorm3D(num_features = num_filters_conv1),
            nn.Conv3d(in_channels = num_filters_conv1, out_channels = num_filters_conv2, kernel_size = (3,3,3), padding='same'),
            nn.BatchNorm3D(num_features= num_filters_conv2),
            nn.ReLu()
            )
        
    def forward(self,x):
        return self.double_conv3D(x)
    
    
class DownSampling(nn.Module):
    def __init__(self):
        self.down = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
    
    def forward(self,x):
        return self.down(x)
        
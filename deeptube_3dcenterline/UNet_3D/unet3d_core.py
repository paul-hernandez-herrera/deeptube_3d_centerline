from torch import nn, tensor, cat

######################################################################        
class Double_convolution_3D(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2):
        super().__init__()
        
        self.double_conv3D = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = num_filters_conv1, kernel_size = (3,3,3), padding= 1),
            nn.BatchNorm3d(num_features = num_filters_conv1),
            nn.ReLU(),
            nn.Conv3d(in_channels = num_filters_conv1, out_channels = num_filters_conv2, kernel_size = (3,3,3), padding= 1),
            nn.BatchNorm3d(num_features= num_filters_conv2),
            nn.ReLU()
            )
        
    def forward(self,x):
        return self.double_conv3D(x)

######################################################################    
class UpSampling_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = (2,2,2), stride=2)
    
    def forward(self, x):
        return self.up(x)
    
###################################################################### 
def concatenate(x1, x2):
    #we have a 3d tensor (N, C, D, H, W)
    #x2 must be the upsampled vector
    
    pad_size = tensor(x1.shape[2:5]) - tensor(x2.shape[2:5])
    half_pad_size = pad_size//2
    
    m = nn.ReflectionPad3d((half_pad_size[2],pad_size[2]-half_pad_size[2], 
                     half_pad_size[1],pad_size[1]-half_pad_size[1],
                     half_pad_size[0],pad_size[0]-half_pad_size[0]))
    x2 = m(x2)
    
    return cat((x1,x2),dim=1)
    

######################################################################    
class Encoder_3D(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2):
        super().__init__()
        self.double_conv3D = Double_convolution_3D(in_channels, num_filters_conv1, num_filters_conv2)
        self.down = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
    
    def forward(self,x):
        x_conv = self.double_conv3D(x)
        x_conv_down = self.down(x_conv)
        return x_conv, x_conv_down


######################################################################    
class Decoder_3D(nn.Module):
    def __init__(self, in_channels, num_filters_conv1, num_filters_conv2):
        super().__init__()
        self.double_conv3D = Double_convolution_3D(in_channels, num_filters_conv1, num_filters_conv2)
        
        self.upsampling = UpSampling_3D(in_channels, in_channels//2)
        
    def forward(self, x_encoder, x_conv):
        
        x_up = self.upsampling(x_conv)
        x = concatenate(x_encoder, x_up)
        return self.double_conv3D(x)



    
    
    

    
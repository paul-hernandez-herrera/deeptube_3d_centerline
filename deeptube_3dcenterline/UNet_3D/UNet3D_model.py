import torch.nn as nn
from  . import unet3d_core 

class Classic_UNet_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #this block allows to initialize the parameters of the network
        self.encoder1 = unet3d_core.Encoder_3D(in_channels, 32, 32)
        self.encoder2 = unet3d_core.Encoder_3D(32, 64, 64)
        self.encoder3 = unet3d_core.Encoder_3D(64, 128, 128)
        self.encoder4 = unet3d_core.Encoder_3D(128, 256, 256)
        
        self.encoder_bottom = unet3d_core.Double_convolution_3D(256, 512, 512)
        
        self.decoder4 = unet3d_core.Decoder_3D(256 + 256, 256, 256)
        self.decoder3 = unet3d_core.Decoder_3D(128 + 128, 128, 128)
        self.decoder2 = unet3d_core.Decoder_3D(64 + 64, 64, 64)
        self.decoder1 = unet3d_core.Decoder_3D(32 + 32, 32, 32)  
        self.outConv  = nn.Conv3d(32, out_channels, kernel_size = (1,1,1))
        
        
    def forward(self,x):
        
        out_encoder1, x = self.encoder1(x)
        out_encoder2, x = self.encoder2(x)
        out_encoder3, x = self.encoder3(x)
        out_encoder4, x = self.encoder4(x)
        bottom_conv = self.encoder_bottom(x)
        x = self.decoder4(out_encoder4, bottom_conv)
        x = self.decoder3(out_encoder3, x)
        x = self.decoder2(out_encoder2, x)
        x = self.decoder1(out_encoder1, x)
        x = self.outConv(x)
        return x
        

        
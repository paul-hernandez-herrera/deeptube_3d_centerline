import torch

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, model_output, target):
        # this function computer the dice loss function for two tensor. 
        # It assumes that model_output and target are of the same size [B, C, D1, D2, ..., DN]
        # B and C are batch size and number of channels. D1, D2, ..., DN are additional dimentions, 
        # for 2D images D1, D2 are widht and height, while for 3D images is depth, widht, height 
        # We assume that the model returns the is not normalized to probabilities [0,1].
        
        print(model_output.shape)
        print(target.shape)
        # Normalizing to [0,1]
        output_normalized_0_1 = torch.sigmoid(model_output)
        
        # convert to 1-d vector
        output_normalized_0_1 = output_normalized_0_1.view(-1)
        target = target.view(-1)
        
        # calculating the metrics over vectors in general terms
        intersection = (output_normalized_0_1 * target).sum() 
        union = output_normalized_0_1.sum() + target.sum() 
        
        if union==0:
            dice = 1
        else:
            dice = 2*intersection/union 
        
        # goal minimize the metric. Dice best performance is at maximum value equal to one, then substracting one
        return 1-dice
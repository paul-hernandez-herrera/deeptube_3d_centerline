class Train_Model():
    def __init__(self, model,
                 train_loader,
                 criterion,
                 device, 
                 optimizer,
                 n_epochs):
        
        self.model = model
        self.train_loader = train_loader
        self.device
        self.n_epochs
        self.optimizer = optimizer
        
    def run(self):
        for epoch in range(0, self.n_epochs):
            self.model.train() #set the model in training mode
            
            for batch_iter in self.train_loader: #update weight for each batch of images
                imgs, targets = batch_iter
                
                network_output = self.model(imgs) #apply network to batch of images
                
                #compute the error between network_output and targets
                loss = self.criterion(network_output, targets)
                
                self.optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensor s to zero.
                
                loss.backward() #compute the gradients given the loss value
                
                self.optimizer.step() #update the weights using the gradients and the approach by optimizer
                
        
    
if __name__=='__main__':
    print('Running from command line to finish')
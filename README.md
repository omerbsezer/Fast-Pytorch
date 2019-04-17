# Fast-Pytorch

This repo aims to cover Pytorch details, Pytorch example implementations, Pytorch sample codes, running Pytorch codes with Google Colab.

## Table of Contents:
- Pytorch Tutorial
  - What is Pytorch?
  - Pytorch Dynamic Graph
  - Pytorch PlayGround
  - Pytorch Cheatsheet
- Pytorch with Google Colab
  - Using Drive
  - Transfer from Github to Colab
- Pytorch Example Implementations
  - MLP (classification)
  - MLP (regression)
  - CNN, 
  - LSTM, 
  - GRU,
  - Transfer Learning
  - DCGAN, 
  - ChatBot
- Pytorch Sample Codes
  - CycleGAN [[github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [[github2]](https://github.com/znxlwm/pytorch-CycleGAN)
  - [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)
  - wiseodd/generative-models, both pytorch and tensorflow [[github]](https://github.com/wiseodd/generative-models)
  - GAN, LSGAN, WGAN, DRAGAN, CGAN, infoGAN, ACGAN, EBGAN, BEGAN [[github]](https://github.com/znxlwm/pytorch-generative-model-collections)
  - CartoonGAN [github](https://github.com/znxlwm/pytorch-CartoonGAN)
  - Pix2Pix [[github]](https://github.com/znxlwm/pytorch-pix2pix), [[paper]]()
  
## Pytorch Tutorial

### What is Pytorch (torch)?
It's python deep learning framework/library that is developed by Facebook. 
 - [What is Pytorch?](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
Pytorch has own datastructure that provides automatic differentiation for all operations on Tensors. 
 - [Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
 
Important keys: torch.Tensor, .requires_grad, .backward(), .grad, with torch.no_grad().
- Pytorch Playground: [Notebook]

Neural Nettwork (nn):

Optimizer:

Loss:

Basic two layer feed forward neural networks with optimizer, loss:
```
import torch
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
        
N_batchsize, D_input, Hidden_size D_output = 64, 1000, 100, 10
epoch=500

x = torch.randn(N_batchsize, D_input)
y = torch.randn(N_batchsize, D_output)

model = TwoLayerNet(D_input, Hidden, D_output)

criterion = torch.nn.MSELoss(reduction='sum') # loss, mean square error 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) # optimizer, stochastic gradient descent, lr=learning rate

for t in range(epoch):
    y_pred = model(x) # Forward pass
    loss = criterion(y_pred, y) #print(t, loss.item())
    optimizer.zero_grad() # Zero gradients,
    loss.backward() # backward pass
    optimizer.step() # update the weights
``` 
 ### What is torchvision?
 
 #### Datasets:

  ## Pytorch Dynamic Graph
  ## Pytorch PlayGround
  ## Pytorch Cheatsheet

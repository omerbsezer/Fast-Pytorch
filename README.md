# Fast-Pytorch

This repo aims to cover Pytorch details, Pytorch example implementations, Pytorch sample codes, running Pytorch codes with Google Colab in a nutshell.

## Table of Contents:
- Pytorch Tutorial
- Pytorch with Google Colab
- Pytorch Example Implementations
- Pytorch Sample Codes
  - CycleGAN [[github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [[github2]](https://github.com/znxlwm/pytorch-CycleGAN)
  - [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)
  - wiseodd/generative-models, both pytorch and tensorflow [[github]](https://github.com/wiseodd/generative-models)
  - GAN, LSGAN, WGAN, DRAGAN, CGAN, infoGAN, ACGAN, EBGAN, BEGAN [[github]](https://github.com/znxlwm/pytorch-generative-model-collections)
  - CartoonGAN [github](https://github.com/znxlwm/pytorch-CartoonGAN)
  - Pix2Pix [[github]](https://github.com/znxlwm/pytorch-pix2pix), [[paper]]()
  
## Pytorch Tutorial

It's python deep learning framework/library that is developed by Facebook. Pytorch has own datastructure that provides automatic differentiation for all operations on Tensors. 
 - [What is Pytorch?](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
 - [Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
 - [Details - Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
 
**Important keys:** torch.Tensor, .requires_grad, .backward(), .grad, with torch.no_grad().
- Pytorch Playground: [Notebook]

**Model (Neural Network,nn):**
```Python
```
**Optimizer:**  [Details](https://pytorch.org/docs/stable/optim.html)
```Python
torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False) # stochastic gradient descent
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```
**Loss Functions:** [Details](https://pytorch.org/docs/stable/nn.html#loss-functions)
```Python
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean') # L1 Loss
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean') # Mean square error loss
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
torch.nn.CTCLoss(blank=0, reduction='mean') #Connectionist Temporal Classification loss
torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean') #negative log likelihood loss
torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean') # Kullback-Leibler divergence Loss
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean') # Binary Cross Entropy
torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```
**Pooling Layers:** [Details](https://pytorch.org/docs/stable/nn.html#pooling-layers)
```Python
```
**Non-linear activation functions:** [Details](https://pytorch.org/docs/stable/nn.html#non-linear-activation-functions)
```Python
```
**Basic two layer feed forward neural networks with optimizer, loss:**
```Python
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

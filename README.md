# Fast-Pytorch

This repo aims to cover Pytorch details, Pytorch example implementations, Pytorch sample codes, running Pytorch codes with Google Colab in a nutshell.

## Table of Contents:
- Fast Pytorch Tutorial
- Fast Torchvision Tutorial
- Pytorch with Google Colab
- Pytorch Example Implementations
- Pytorch Sample Codes
  - CycleGAN [[github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [[github2]](https://github.com/znxlwm/pytorch-CycleGAN)
  - [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)
  - wiseodd/generative-models, both pytorch and tensorflow [[github]](https://github.com/wiseodd/generative-models)
  - GAN, LSGAN, WGAN, DRAGAN, CGAN, infoGAN, ACGAN, EBGAN, BEGAN [[github]](https://github.com/znxlwm/pytorch-generative-model-collections)
  - CartoonGAN [github](https://github.com/znxlwm/pytorch-CartoonGAN)
  - Pix2Pix [[github]](https://github.com/znxlwm/pytorch-pix2pix), [[paper]]()
  
## Fast Pytorch Tutorial

It's python deep learning framework/library that is developed by Facebook. Pytorch has own datastructure that provides automatic differentiation for all operations on Tensors. 
 - [What is Pytorch?](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
 - [Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
 - [Details - Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
 
**Important keys:** torch.Tensor, .requires_grad, .backward(), .grad, with torch.no_grad().
- Pytorch Playground: [Notebook]

**Model (Neural Network Layers:** [Details](https://pytorch.org/docs/stable/nn.html)
```Python
torch.nn.RNN(*args, **kwargs)
torch.nn.LSTM(*args, **kwargs)
torch.nn.GRU(*args, **kwargs)
torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
torch.nn.LSTMCell(input_size, hidden_size, bias=True)
torch.nn.GRUCell(input_size, hidden_size, bias=True)
torch.nn.Linear(in_features, out_features, bias=True)
torch.nn.Bilinear(in1_features, in2_features, out_features, bias=True)
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)
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
torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0) # Computes a partial inverse of MaxPool2d
torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
torch.nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
torch.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False) # 2D power-average pooling 
torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
torch.nn.AdaptiveAvgPool2d(output_size)
```
**Non-linear activation functions:** [Details](https://pytorch.org/docs/stable/nn.html#non-linear-activation-functions)
```Python
torch.nn.ELU(alpha=1.0, inplace=False) #  the element-wise function
torch.nn.Hardshrink(lambd=0.5) #  hard shrinkage function element-wise
torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
torch.nn.PReLU(num_parameters=1, init=0.25)
torch.nn.ReLU(inplace=False)
torch.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False) # randomized leaky rectified liner unit function
torch.nn.SELU(inplace=False)
torch.nn.CELU(alpha=1.0, inplace=False)
torch.nn.Sigmoid()
torch.nn.Softplus(beta=1, threshold=20)
torch.nn.Softshrink(lambd=0.5)
torch.nn.Tanh()
torch.nn.Tanhshrink()
torch.nn.Threshold(threshold, value, inplace=False)
torch.nn.Softmax(dim=None)
torch.nn.Softmax2d()
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
 ### Fast Torchvision Tutorial
"The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision."
**ImageFolder:** If you have special/custom datasets, image folder function can be used.
```Python
# Example
imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
 ``` 
**Transforms:** Transforms are common for image transformations. [Details](https://pytorch.org/docs/stable/torchvision/transforms.html)
```Python
# Some of the important functions:
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) 3 Example
torchvision.transforms.CenterCrop(size)
torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
torchvision.transforms.Grayscale(num_output_channels=1)
torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
torchvision.transforms.RandomApply(transforms, p=0.5)
torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
torchvision.transforms.RandomGrayscale(p=0.1)
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
torchvision.transforms.RandomVerticalFlip(p=0.5)
torchvision.transforms.Resize(size, interpolation=2)
torchvision.transforms.Scale(*args, **kwargs)
torchvision.transforms.LinearTransformation(transformation_matrix)
torchvision.transforms.Normalize(mean, std, inplace=False) # Normalize a tensor image with mean and standard deviation. 
torchvision.transforms.ToTensor() # Convert a PIL Image or numpy.ndarray to tensor
# Functional transforms give you fine-grained control of the transformation pipeline. As opposed to the transformations above, functional transforms don’t contain a random number generator for their parameters. That means you have to specify/generate all parameters, but you can reuse the functional transform.
torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
torchvision.transforms.functional.hflip(img)
torchvision.transforms.functional.normalize(tensor, mean, std, inplace=False) # Normalize a tensor image with mean and standard deviation
torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None) # Rotate the image by angle
torchvision.transforms.functional.to_grayscale(img, num_output_channels=1) # Convert image to grayscale version of image.
```
**Datasets:**: Most used datasets in the literature. [Details](https://pytorch.org/docs/stable/torchvision/datasets.html)
```Python
torchvision.datasets.MNIST(root='data/mnist', train=True, transform=transform, target_transform=None, download=True) # with example
torchvision.datasets.FashionMNIST(root='data/fashion-mnist', train=True, transform=transform, target_transform=None, download=True) # with example
torchvision.datasets.KMNIST(root, train=True, transform=None, target_transform=None, download=False)
torchvision.datasets.EMNIST(root, split, **kwargs)
torchvision.datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=10, transform=None, target_transform=None, random_offset=0)
torchvision.datasets.CocoCaptions(root, annFile, transform=None, target_transform=None)
torchvision.datasets.CocoDetection(root, annFile, transform=None, target_transform=None)
torchvision.datasets.LSUN(root, classes='train', transform=None, target_transform=None)
torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
torchvision.datasets.STL10(root, split='train', transform=None, target_transform=None, download=False)
torchvision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)
torchvision.datasets.PhotoTour(root, name, train=True, transform=None, download=False)
torchvision.datasets.SBU(root, transform=None, target_transform=None, download=True)
torchvision.datasets.Flickr8k(root, ann_file, transform=None, target_transform=None)
torchvision.datasets.VOCSegmentation(root, year='2012', image_set='train', download=False, transform=None, target_transform=None)
torchvision.datasets.Cityscapes(root, split='train', mode='fine', target_type='instance', transform=None, target_transform=None)
```   
**Models:** [Details](https://pytorch.org/docs/stable/torchvision/models.html)
```Python
# model with random weights
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
# with pre-trained models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
```
**Utils:**
```Python
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0) # Make a grid of images.
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0) # Save a given Tensor into an image file
```
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

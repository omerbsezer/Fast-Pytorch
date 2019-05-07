# Fast-Pytorch

This repo aims to cover Pytorch details, Pytorch example implementations, Pytorch sample codes, running Pytorch codes with Google Colab in a nutshell.

## Running in Colab
- Two way:
   - Clone or download all repo, then upload your drive root file ('/drive/'), open .ipynb files with 'Colaboratory' application
   - Download "Github2Drive.ipynb" and copy your drive root file, open with 'Colaboratory' and run 3 cells one by one, hence repo is cloned to your drive file. ([Pytorch with Google Colab](#pytorchcolab))

## Table of Contents:
- :fire:[Fast Pytorch Tutorial](#pytorchtutorial)
  - [Pytorch Playground](#pytorchplayground)
    - :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Pytorch_Playground.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Pytorch_Playground.ipynb)
  - [Model (Neural Network Layers)](#model)
  - [Optimizer](#optimizer)
  - [Loss Functions](#lossfunctions)
  - [Pooling Layers](#poolinglayers)
  - [Non-linear activation functions](#nonlinearactivation)
  - [Basic 2 Layer NN](#example)
- :fire:[Fast Torchvision Tutorial](#torchvisiontutorial)
  - [ImageFolder](#imagefolder)
  - [Transforms](#transforms)
  - [Datasets](#datasets)
  - [Models](#torchvisionmodels)
  - [Utils](#utils)
- :fire:[Pytorch with Google Colab](#pytorchcolab)
- :fire:[Pytorch Example Implementations](#pytorchexamples)
  - [MLP](#mlp) 
    - MLP 1 Class with Binary Cross Entropy (BCE) Loss: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_1class_BinaryCrossEntropyLoss.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_1class_BinaryCrossEntropyLoss.ipynb) 
    - MLP 2 Classes with Cross Entropy Loss: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_2class_CrossEntropyLoss.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_2class_CrossEntropyLoss.ipynb)
    - MLP 3-Layer with MNIST Example: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_3layer_MNIST.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_3layer_MNIST.ipynb)
   - [CNN](#cnn)
      - CNN with MNIST Example:  :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Mnist.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Mnist.ipynb)
      - Improved CNN with MNIST Example: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Improved_CNN_Mnist.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Improved_CNN_Mnist.ipynb)
    - [CNN Visualization](#cnnvisualization)
      - CNN Visualization:  :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Visualization.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Visualization.ipynb)
    - [RNN](#rnn)
      - RNN Text Generation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/RNN_word_embeddings.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/RNN_word_embeddings.ipynb)
    - [Transfer Learning](#transferlearning)
      - Transfer Learning Implementation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/TransferLearning.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/TransferLearning.ipynb)
    - [DCGAN](#dcgan)
      - DCGAN Implementation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/DCGAN.ipynb),  :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/DCGAN.ipynb)
    - [ChatBot](#chatbot)
      - Chatbot Implementation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/ChatBot.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/ChatBot.ipynb)
- :fire:[Pytorch Sample Codes](#pytorchsamplecodes)
  
## Fast Pytorch Tutorial <a name="pytorchtutorial"></a>

It's python deep learning framework/library that is developed by Facebook. Pytorch has own datastructure that provides automatic differentiation for all operations on Tensors. 
 - [What is Pytorch?](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
 - [Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
 - [Details - Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
 
**Important keys:** torch.Tensor, .requires_grad, .backward(), .grad, with torch.no_grad().

**Pytorch CheatSheet:** :fire:[Details](https://pytorch.org/tutorials/beginner/ptcheat.html)

### Pytorch Playground <a name="pytorchplayground"></a>
- :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Pytorch_Playground.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Pytorch_Playground.ipynb)

### Model (Neural Network Layers) <a name="model"></a>
- :fire:[Details](https://pytorch.org/docs/stable/nn.html)
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
### Optimizer <a name="optimizer"></a>
- :fire:[Details](https://pytorch.org/docs/stable/optim.html)
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
### Loss Functions <a name="lossfunctions"></a>
- :fire:[Details](https://pytorch.org/docs/stable/nn.html#loss-functions)
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
### Pooling Layers <a name="poolinglayers"></a>
- :fire:[Details](https://pytorch.org/docs/stable/nn.html#pooling-layers)
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
### Non-linear activation functions <a name="nonlinearactivation"></a>
- :fire:[Details](https://pytorch.org/docs/stable/nn.html#non-linear-activation-functions)
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
### Basic 2 Layer NN <a name="example"></a>
- Basic two layer feed forward neural networks with optimizer, loss:
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
 ### Fast Torchvision Tutorial <a name="torchvisiontutorial"></a>
"The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision."
### ImageFolder <a name="imagefolder"></a>
- If you have special/custom datasets, image folder function can be used.
```Python
# Example
imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
 ``` 
### Transforms <a name="transforms"></a>
- Transforms are common for image transformations. :fire:[Details](https://pytorch.org/docs/stable/torchvision/transforms.html)
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
### Datasets <a name="datasets"></a>
- Most used datasets in the literature. :fire:[Details](https://pytorch.org/docs/stable/torchvision/datasets.html)
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
### Models <a name="torchvisionmodels"></a>
- :fire:[Details](https://pytorch.org/docs/stable/torchvision/models.html)
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
### Utils <a name="utils"></a>
```Python
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0) # Make a grid of images.
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0) # Save a given Tensor into an image file
```

  
## Pytorch with Google Colab <a name="pytorchcolab"></a>
- If you want to use drive.google for storage, you have to run the following codes for authentication. After running cell, links for authentication are appereared, click and copy the token pass for that session.
```Script
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```
- Then, you can use your drive file and reach the your codes which are in your drive. 
```Script
!mkdir -p drive
!google-drive-ocamlfuse drive
import sys
sys.path.insert(0,'drive/Fast-Pytorch/Learning_Pytorch') # Example, your drive root: 'drive/'
!ls drive
```
- After authentication, git clone command is also used to clone project.  
```Script
%cd 'drive/Fast-Pytorch'
!ls
!git clone https://github.com/omerbsezer/Fast-Pytorch.git
```
## Pytorch Example Implementations <a name="pytorchexamples"></a>
- All codes are run on the Colab. You can also run on desktop jupyter notebooks.(Anaconda)[https://www.anaconda.com/distribution/].
### MLP <a name="mlp"></a> 
- MLP 1 Class with Binary Cross Entropy (BCE) Loss: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_1class_BinaryCrossEntropyLoss.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_1class_BinaryCrossEntropyLoss.ipynb) 
- MLP 2 Classes with Cross Entropy Loss: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_2class_CrossEntropyLoss.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_2class_CrossEntropyLoss.ipynb)
- MLP 3-Layer with MNIST Example: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_3layer_MNIST.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_3layer_MNIST.ipynb)

```Python
class Model(nn.Module):
  def __init__(self):
    super(Model,self).__init__()
    self.fc1 =torch.nn.Linear(x.shape[1],5)
    self.fc2 =torch.nn.Linear(5,3)
    self.fc3 =torch.nn.Linear(3,1)
    self.sigmoid=torch.nn.Sigmoid()
    
  def forward(self,x):
    out =self.fc1(x)
    out =self.sigmoid(out)
    out =self.fc2(out)
    out =self.sigmoid(out)
    out =self.fc3(out)
    out= self.sigmoid(out)
    return out
```
### CNN <a name="cnn"></a>
- CNN with MNIST Example:  :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Mnist.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Mnist.ipynb)
- Improved CNN with MNIST Example: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Improved_CNN_Mnist.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/Improved_CNN_Mnist.ipynb)

```Python
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    # input_size:28, same_padding=(filter_size-1)/2, 3-1/2=1:padding
    self.cnn1=nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    # input_size-filter_size +2(padding)/stride + 1 = 28-3+2(1)/1+1=28
    self.batchnorm1=nn.BatchNorm2d(8)
    # output_channel:8, batch(8)
    self.relu=nn.ReLU()
    self.maxpool1=nn.MaxPool2d(kernel_size=2)
    #input_size=28/2=14
    self.cnn2=nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
    # same_padding: (5-1)/2=2:padding_size. 
    self.batchnorm2=nn.BatchNorm2d(32)
    self.maxpool2=nn.MaxPool2d(kernel_size=2)
    # input_size=14/2=7
    # 32x7x7=1568
    self.fc1 =nn.Linear(in_features=1568, out_features=600)
    self.dropout= nn.Dropout(p=0.5)
    self.fc2 =nn.Linear(in_features=600, out_features=10)
  def forward(self,x):
    out =self.cnn1(x)
    out =self.batchnorm1(out)
    out =self.relu(out)
    out =self.maxpool1(out)
    out =self.cnn2(out)
    out =self.batchnorm2(out)
    out =self.relu(out)
    out =self.maxpool2(out)
    out =out.view(-1,1568)
    out =self.fc1(out)
    out =self.relu(out)
    out =self.dropout(out)
    out =self.fc2(out)
    return out
```    
### CNN Visualization <a name="cnnvisualization"></a>
- CNN Visualization:  :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Visualization.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Visualization.ipynb)

![visualization-CNN-runtime](https://user-images.githubusercontent.com/10358317/57305262-6d99d600-70e9-11e9-9a8f-7f9ea0f69dc3.png)

### RNN <a name="rnn"></a>
- RNN Text Generation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/RNN_word_embeddings.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/RNN_word_embeddings.ipynb)

```Python
class TextGenerator(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
    super(TextGenerator, self).__init__()
    self.embed= nn.Embedding(vocab_size,embed_size)
    self.lstm=nn.LSTM(embed_size,hidden_size,num_layers, batch_first=True)
    self.linear=nn.Linear(hidden_size, vocab_size)
    
  def forward(self,x,h):
    x= self.embed(x)
    # h: hidden_state, c=output
    # x= x.view(batch_size,timesteps,embed_size)
    out, (h,c)=self.lstm(x,h)
    #(batch_size*timesteps, hidden_size)
    #out.size(0):batch_size; out.size(1):timesteps, out.size(2): hidden_size
    out=out.reshape(out.size(0)*out.size(1),out.size(2))
    # decode hidden states of all time steps
    out= self.linear(out)
    return out, (h,c)
```   
### Transfer Learning <a name="transferlearning"></a>

- Transfer Learning Implementation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/TransferLearning.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/TransferLearning.ipynb)

![transferlearning](https://user-images.githubusercontent.com/10358317/57308748-8efdc080-70ef-11e9-8bca-68d672d5dde6.jpg)

### DCGAN <a name="dcgan"></a>

- DCGAN Implementation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/DCGAN.ipynb),  :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/DCGAN.ipynb)

![dcgan](https://user-images.githubusercontent.com/10358317/57308734-8ad1a300-70ef-11e9-830e-8a8b04d7b00d.png)

### ChatBot <a name="chatbot"></a>
- Chatbot Implementation: :green_book:[[Colab]](https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/ChatBot.ipynb), :notebook:[[Notebook]](https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/ChatBot.ipynb)
- Chatbot implementation [details](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).

```Script
> what is your name?
Bot: berger .
> are you married?
Bot: no .
> how old are you?
Bot: i m not hungry .
> how are you?
Bot: okay .
> where are you from?
Bot: i m travelling .
> do you know me?
Bot: yes .
> who am i?
Bot: i don t know .
> what is your job?
Bot: i m not going to tell you .
> what is your problem?
Bot: i m not afraid of anything .
> are you robot?
Error: Encountered unknown word.
> what is my name?
Bot: berger .
> ai?
Error: Encountered unknown word.
> what do you want to me?
Bot: i m going to kill you .
> how do you kill me?
Bot: i told you .
> what is your plan?
Bot: i m not going to tell you .
> are you live?
Bot: yes .
> where?
Bot: the zoo .
> what is zoo?
Bot: the sheets . . .
> where is the zoo?
Bot: i don t know .
``` 

## Pytorch Sample Codes <a name="pytorchsamplecodes"></a>
  - CycleGAN [[github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [[github2]](https://github.com/znxlwm/pytorch-CycleGAN)
  - [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)
  - Wiseodd/generative-models, both pytorch and tensorflow [[github]](https://github.com/wiseodd/generative-models)
  - GAN, LSGAN, WGAN, DRAGAN, CGAN, infoGAN, ACGAN, EBGAN, BEGAN [[github]](https://github.com/znxlwm/pytorch-generative-model-collections)
  - CartoonGAN [github](https://github.com/znxlwm/pytorch-CartoonGAN)
  - Pix2Pix [[github]](https://github.com/znxlwm/pytorch-pix2pix)

## References <a name="references"></a>

- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- [Udemy Course: the-complete-neural-networks-bootcamp-theory-applications](https://www.udemy.com/the-complete-neural-networks-bootcamp-theory-applications/)

# GAN-for-Nature-Images
This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) composed of a Discriminator and a Generator. GANs are a type of deep learning model that consists of two neural networks competing against each other: the Generator generates data samples, and the Discriminator tries to differentiate between real and generated samples. This adversarial setup leads to the training of a generator that produces realistic-looking data samples.

## Discriminator

The `Discriminator` class is responsible for distinguishing between real and generated data samples. It's implemented as a feedforward neural network with the following architecture:

- Input Layer: Accepts data samples with dimensions `img_dim`.
- Hidden Layer 1: A fully connected layer with 128 neurons, followed by a Leaky ReLU activation function (slope 0.1).
- Hidden Layer 2: Another fully connected layer with 1 neuron, followed by a Sigmoid activation function.
- Output Layer: Produces a probability score indicating the likelihood that the input is a real sample.

The `forward` method computes the discriminator's output for a given input data sample.

```python
 class Discriminator(nn.Module): ''' Discriminator Class Values: im_chan: the number of channels in the images, fitted for the dataset used, a scalar (MNIST is black-and-white, so 1 channel is your default) hidden_dim: the inner dimension, a scalar ''' def __init__(self, im_chan=3, hidden_dim=256, output_dim=(1, 4, 4)): super(Discriminator, self).__init__() self.disc = nn.Sequential( self.make_disc_block(im_chan, hidden_dim), self.make_disc_block(hidden_dim, hidden_dim * 2), self.make_disc_block(hidden_dim * 2, output_dim[0], final_layer=True), ) def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False): ''' Function to return a sequence of operations corresponding to a discriminator block of DCGAN, corresponding to a convolution, a batchnorm (except for in the last layer), and an activation. Parameters: input_channels: how many channels the input feature representation has output_channels: how many channels the output feature representation should have kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size) stride: the stride of the convolution final_layer: a boolean, true if it is the final layer and false otherwise (affects activation and batchnorm) ''' # Build the neural block if not final_layer: return nn.Sequential( nn.Conv2d(input_channels, output_channels, kernel_size, stride), nn.BatchNorm2d(output_channels), nn.LeakyReLU(0.2) ) else: # Final Layer return nn.Sequential( nn.Conv2d(input_channels, output_channels, kernel_size, stride), ) def forward(self, image): ''' Function for completing a forward pass of the discriminator: Given an image tensor, returns a 1-dimension tensor representing fake/real. Parameters: image: a flattened image tensor with dimension (im_dim) ''' disc_pred = self.disc(image) return disc_pred.view(len(disc_pred), -1)
```

## Generator

The `Generator` class is responsible for generating synthetic data samples that resemble real data. It's implemented as a feedforward neural network with the following architecture:


- Input Layer: Accepts a random noise vector of dimension `z_dim`.
- Hidden Layer 1: A fully connected layer with 512 neurons, followed by a Leaky ReLU activation function (slope 0.1).
- Hidden Layer 2: Another fully connected layer with 1024 neurons, followed by a Leaky ReLU activation function.
- Output Layer: Produces synthetic data samples with dimensions `img_dim`, using the Tanh activation function to ensure values are in the range [-1, 1].

The `forward` method generates synthetic data samples from a given noise vector.

```python
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, img_dim),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.gen(x)

```
## Usage

To use the Discriminator and Generator classes in your project, follow these steps:

1. Install the required dependencies, preferably using a virtual environment.
2. Import the necessary modules: `import torch.nn as nn`.
3. Create instances of the Discriminator and Generator classes, providing the required input dimensions (`img_dim` and `z_dim`).
4. Use the `forward` method of the Generator to generate synthetic data samples.
5. Use the Discriminator to evaluate the likelihood of a given data sample being real or fake.

Example code snippet:

```python
import torch.nn as nn

# Define the dimensions
img_dim = ...
z_dim = ...

# Instantiate the Discriminator and Generator
discriminator = Discriminator(img_dim)
generator = Generator(z_dim, img_dim)

# Generate synthetic data
noise = torch.randn(batch_size, z_dim)
generated_data = generator(noise)

# Evaluate data using the Discriminator
real_data = ...
discriminator_output_real = discriminator(real_data)
discriminator_output_fake = discriminator(generated_data)
```

## Acknowledgments

This implementation was inspired by the concepts of Generative Adversarial Networks (GANs) and the PyTorch library.

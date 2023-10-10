# GAN Project Readme

## Introduction

This repository contains the code and information for a Generative Adversarial Network (GAN) project. In this project, a GAN model is implemented using PyTorch to generate images based on a dataset of 90,000 1024x1024 images from the [LHQ-1024 dataset](https://www.kaggle.com/datasets/dimensi0n/lhq-1024). The GAN consists of a Generator and a Discriminator neural network. The goal is to train the Generator to generate realistic images that are indistinguishable from the images in the dataset, while the Discriminator aims to distinguish between real and fake images.

## Requirements

To run this project, you will need the following Python libraries:

- PyTorch
- Matplotlib
- NumPy
- torchvision

You can install these libraries using pip:

```bash
pip install torch matplotlib numpy torchvision
```

## Dataset

The dataset used for this project is the LHQ-1024 dataset, which contains 90,000 high-resolution (1024x1024) images. You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/dimensi0n/lhq-1024) and organize it in your preferred directory structure.

## Discriminator

The Discriminator is a convolutional neural network responsible for distinguishing between real and fake images. It consists of multiple convolutional layers followed by batch normalization and leaky ReLU activations. The final layer produces a 1-dimensional tensor representing the probability that the input image is real.

The Discriminator class is defined as follows:

```python
class Discriminator(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=256, output_dim=(1, 4, 4)):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, output_dim[0], final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
```

## Generator

The Generator is another convolutional neural network responsible for generating fake images from random noise. It takes a random noise vector as input and progressively upsamples it to produce a fake image. The final layer uses a hyperbolic tangent (tanh) activation function to ensure that the generated image pixels are within the range [-1, 1].

The Generator class is defined as follows:

```python
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)
```

## Training Parameters

The following training parameters are used in this project:

- Batch Size: 32
- Loss Function: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss()`)

You can adjust these parameters according to your specific requirements or hardware capabilities.

## Usage

To train the GAN model, you can use the provided code and dataset. Be sure to set up your environment with the required libraries and download the LHQ-1024 dataset as mentioned above. After that, you can run the training script to train the GAN.

Example training script (not provided):

```python
# Import necessary libraries
import torch
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator
from dataset import CustomDataset
from torch.utils.data import DataLoader
# ...

# Initialize the Generator and Discriminator
generator = Generator(z_dim=100, im_chan=3, hidden_dim=64)
discriminator = Discriminator(im_chan=3, hidden_dim=64)

# Define loss function and optimizers
criterion = nn.BCEWithLogitsLoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Create DataLoader for your dataset
# dataset = CustomDataset(...)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for real_images in data_loader:
        # Train Discriminator
        # ...

        # Train Generator
        # ...

    # Save generated images and model checkpoints
    # ...

# ...
```

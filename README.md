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
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
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

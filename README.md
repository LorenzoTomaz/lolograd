# LoloGrad: A Simple Neural Network Framework with Autograd

## Introduction

LoloGrad is a minimalist neural network framework designed for educational purposes. It focuses on the core concepts of automatic differentiation (autograd) and backpropagation, making it an excellent tool for learning and experimentation. LoloGrad is particularly suitable for those looking to understand the internals of neural network operations and autograd mechanisms.

## Features

Automatic Differentiation (Autograd): At the heart of LoloGrad is an autograd system that automatically computes gradients for node operations. This feature simplifies the implementation of backpropagation, allowing users to focus on designing and training neural networks without worrying about the complex underlying calculus.

- Dense Layers: LoloGrad comes with fully implemented dense layers (also known as fully connected layers). These layers are fundamental building blocks of many neural networks, suitable for a wide range of applications from simple regression to complex classification tasks.

- Convolutional 2D Layers: In addition to dense layers, LoloGrad offers 2D convolutional layers out of the box. These layers are crucial for tasks involving spatial data, such as image and video recognition, making LoloGrad suitable for experimenting with convolutional neural networks (CNNs).

## Getting Started

### Prerequisites

- Python 3.10

## Installation

Currently, LoloGrad is not available as a package on PyPI. To use LoloGrad, clone the repository from GitHub:

bash
Copy code
git clone https://github.com/yourusername/lolograd.git
cd lolograd

## Basic Usage

Here's a quick example of how to use LoloGrad to create a simple neural network:

```python
from lolograd.engine import Value
from lolograd.layers import DenseLayer, ConvolutionalLayer
from lolograd.model import NeuralNetwork

from sklearn.datasets import load_iris
from lolograd.examples.mlp import MLP
from lolograd.loss.cross_entropy import CrossEntropyLoss

from lolograd.utils.hot_encode import one_hot_encode


IRIS_DATASET = load_iris()


def load_iris_dataset_classification():
    X = IRIS_DATASET.data
    y = IRIS_DATASET.target
    return X, y


dataset = load_iris_dataset_classification()

X, y = dataset
num_classes = len(set(y))
y = list(map(lambda x: one_hot_encode(x, num_classes), y))
mlp_hidden_dim = 50
model = MLP(X.shape[1], [mlp_hidden_dim, num_classes])
learning_rate = 0.01
loss_function = CrossEntropyLoss()
for epoch in range(10):
    total_loss = 0
    for i in range(0, len(X)):
        x = X[i]
        y_true = y[i]
        y_pred = model(x)
        loss = loss_function(y_pred, y_true)
        total_loss += loss.data
        for p in model.parameters():
            p.grad = 0
        loss.backward()
        for param in model.parameters():
            param.data -= learning_rate * param.grad
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

```

## Contributing

Contributions to LoloGrad are welcome! Whether it's improving the documentation, adding new features, or reporting issues, all contributions are appreciated.

## License

LoloGrad is open source and is available under the MIT License.

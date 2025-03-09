# Neural Network Implementation for MNIST Digit Recognition

A personal project implementing a neural network from scratch in Python for handwritten digit recognition using the MNIST dataset. This implementation focuses on educational purposes and demonstrates the fundamental concepts of deep learning without relying on existing frameworks.

## Overview

This project implements a fully-connected neural network with configurable architecture, supporting key features such as:

- Mini-batch gradient descent with Adam optimization
- Early stopping mechanism for preventing overfitting
- Model persistence (save/load functionality)
- Interactive digit drawing interface for real-time testing

## Technical Details

### Neural Network Architecture

- Input Layer: 784 neurons (28x28 flattened images)
- Hidden Layers: Configurable, default [128, 64]
- Output Layer: 10 neurons (digits 0-9)
- Activation Functions:
  - Hidden Layers: ReLU
  - Output Layer: Softmax

### Implementation Features

- Forward Propagation with ReLU and Softmax activations
- Backpropagation with gradient descent
- Adam optimizer for weight updates
- Mini-batch processing for efficient training
- Best weights saving during training
- Early stopping based on validation loss

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
.venv/bin/activate  # On Unix
# or
.venv\Scripts\activate  # On Windows

# Install dependencies from lock file
uv pip install -r requirements.lock
```

Note: If you want to update the dependencies, you can recompile the requirements:
```bash
uv pip compile requirements.txt -o requirements.lock
```

## Interactive Testing

```python
from Utils.drawing_predictor import launch_drawing_predictor

# Launch drawing interface
launch_drawing_predictor(network)
```

## Performance

The implementation achieves the following performance metrics on the MNIST dataset:
- Training Accuracy: ~98%
- Test Accuracy: ~97%

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Felipe Rocha Martins

## Acknowledgments

- The MNIST dataset providers
- The scientific Python community for their tools and documentation

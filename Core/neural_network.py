import numpy as np
import json
import os

class NeuralNetwork:
    """
    A fully connected neural network implementation from scratch.
    
    This neural network uses ReLU activation for hidden layers and softmax for the output layer.
    It implements mini-batch gradient descent with Adam optimization and supports early stopping.
    
    Attributes:
        layers (list): List of integers representing the number of neurons in each layer
        weights (list): List of weight matrices between layers
        bias (list): List of bias vectors for each layer
        adam (dict): Dictionary containing Adam optimizer state
    """
    
    def __init__(self, input_size=784, hidden_layers=[512,512], output_size=10):
        """
        Initializes a neural network with specified architecture.
        
        Args:
            input_size: Number of input features (default: 784 for MNIST 28x28 images)
            hidden_layers: List of integers specifying the number of neurons in each hidden layer
                         (default: [512,512] for two hidden layers with 512 neurons each)
            output_size: Number of output classes (default: 10 for MNIST digits 0-9)
        
        The network is initialized with small random weights (scaled by 0.01) and zero biases.
        This initialization helps prevent initial saturation of neurons and allows for better
        gradient flow during training.
        
        Example architecture for MNIST:
            Input Layer (784) -> Hidden Layer 1 (512) -> Hidden Layer 2 (512) -> Output Layer (10)
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.bias = []

        # Setup weights and bias for input layer to first hidden layer
        self.weights.append(0.01 * np.random.randn(self.input_size, self.hidden_layers[0]))
        self.bias.append(np.zeros((1, self.hidden_layers[0])))
        
        # Setup weights and bias for hidden layers
        for i in range(len(self.hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1]))
            self.bias.append(np.zeros((1, self.hidden_layers[i+1])))

        # Setup weights and bias for last hidden layer to output layer
        self.weights.append(0.01 * np.random.randn(self.hidden_layers[-1], self.output_size))
        self.bias.append(np.zeros((1, self.output_size)))
        
        # Initialize Adam optimizer state
        self.adam = {
            'weights': [{
                'm': np.zeros_like(w),
                'v': np.zeros_like(w)
            } for w in self.weights],
            'bias': [{
                'm': np.zeros_like(b),
                'v': np.zeros_like(b)
            } for b in self.bias],
            't': 0  # timestep
        }
    
    def forward(self, inputs):
        """
        Performs forward propagation through the neural network. Uses ReLU activation for hidden layers and softmax for output layer.
        
        Args:
            inputs: Input data of shape (batch_size, input_features)
                   batch_size: number of examples in this batch
                   input_features: number of features for each example (e.g., 784 for MNIST)
            
        Returns:
            tuple: (output_probabilities, cache)
                - output_probabilities: Network predictions of shape (batch_size, num_classes)
                - cache: Dictionary containing intermediate values needed for backpropagation:
                    - Z: List of pre-activation values for each layer
                    - A: List of post-activation values for each layer (including input)
        
        Example shapes for MNIST:
            - inputs: (100, 784)        # 100 images, each 28x28 = 784 pixels
            - hidden1: (100, 512)       # First hidden layer with 512 neurons
            - hidden2: (100, 512)       # Second hidden layer with 512 neurons
            - output: (100, 10)         # 10 classes (digits 0-9)
        """
        # Initialize cache dictionary to store values needed for backpropagation
        # We need these values to compute gradients during backward pass
        cache = {
            'Z': [],  # Pre-activation values (before ReLU/softmax)
                     # Z = W @ A + b for each layer
            
            'A': [inputs],  # Post-activation values (after ReLU/softmax)
                           # Starts with input (A0) and stores output of each layer (A1, A2, ...)
        }
        
        # Iterate through each layer of the network
        for i in range(len(self.weights)):
            # Step 1: Linear transformation (Z = W @ A + b)
            # --------------------------------------------
            # weights[i] shape: (prev_layer_size, current_layer_size)
            # cache['A'][-1] shape: (batch_size, prev_layer_size)
            # bias[i] shape: (1, current_layer_size)
            # Result z shape: (batch_size, current_layer_size)
            z = np.dot(cache['A'][-1], self.weights[i]) + self.bias[i]
            cache['Z'].append(z)
            
            # Step 2: Apply activation function
            # --------------------------------
            if i < len(self.weights) - 1:
                # Hidden layers: ReLU activation
                # ReLU(x) = max(0, x)
                # - Keeps positive values unchanged
                # - Sets negative values to 0
                # - Helps network learn non-linear patterns
                a = np.maximum(0, z)
            else:
                # Output layer: Softmax activation
                # softmax(x) = exp(x) / sum(exp(x))
                # 1. Subtract max for numerical stability (prevents exp overflow)
                # 2. Apply exp to get positive values
                # 3. Normalize by sum to get probabilities (sum to 1)
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # Store post-activation values
            # Each A[i] will be used to:
            # 1. Compute next layer's Z value
            # 2. Compute weight gradients during backpropagation
            cache['A'].append(a)
        
        # Return both:
        # 1. Network output (final layer activation)
        # 2. Cache containing all intermediate values for backpropagation
        return cache['A'][-1], cache

    def backward(self, cache, targets):
        """
        Performs backpropagation to compute gradients of the loss with respect to parameters.
        Implements the chain rule to compute gradients layer by layer, from output to input.
        Uses cross-entropy loss combined with softmax activation for classification.
        
        Args:
            cache: Dictionary containing intermediate values from forward pass
                - Z: List of pre-activation values for each layer
                - A: List of post-activation values for each layer
                    A[0] is input, A[1:] are layer outputs
            targets: True labels in one-hot format, shape (batch_size, num_classes)
                    Each row is a one-hot vector where 1 indicates the true class
        
        Returns:
            dict: Gradients for network parameters
                - weights: List of gradients for weight matrices
                - bias: List of gradients for bias vectors
        
        The backward pass follows these steps for each layer:
        1. Compute gradient of loss with respect to layer output (dZ)
        2. Compute gradients for weights: dW = (1/batch_size) * (A_prev.T @ dZ)
        3. Compute gradients for biases: db = (1/batch_size) * sum(dZ)
        4. Compute gradient for next layer: dA = dZ @ W.T
        5. Apply ReLU derivative: dZ = dA * (Z > 0)
        
        Note: The softmax activation and cross-entropy loss are combined in the
        forward pass, so the backward pass only needs to compute the gradient of
        the cross-entropy loss with respect to the pre-activation values.
        """
        batch_size = targets.shape[0]
        num_layers = len(self.weights)
        
        # Initialize gradients dictionary
        grads = {
            'weights': [None] * num_layers,
            'bias': [None] * num_layers,
        }
        
        # Compute initial gradient from softmax + cross-entropy
        # dL/dz = softmax(z) - targets
        dZ = cache['A'][-1] - targets  # Shape: (batch_size, num_classes)
        
        # Iterate backwards through the layers
        for l in reversed(range(num_layers)):
            # Current layer gradients
            # dL/dW = (1/m) * (A_prev.T @ dZ)
            grads['weights'][l] = (1/batch_size) * np.dot(cache['A'][l].T, dZ)
            # dL/db = (1/m) * sum(dZ)
            grads['bias'][l] = (1/batch_size) * np.sum(dZ, axis=0, keepdims=True)
            
            # Compute dZ for previous layer (if not at input layer)
            if l > 0:
                # dZ = (dZ @ W.T) * ReLU'(Z)
                # ReLU'(Z) = 1 if Z > 0, else 0
                dA = np.dot(dZ, self.weights[l].T)
                dZ = dA * (cache['Z'][l-1] > 0)
        
        return grads
    
    def init_adam(self):
        """
        Initializes Adam optimizer parameters for all network weights and biases.
        
        Adam combines the benefits of two other optimizers:
        1. Momentum: Helps accelerate SGD in relevant directions and dampens oscillations
        2. RMSprop: Adapts learning rates based on the history of gradients
        
        For each parameter (weights and biases), we maintain:
        - m: Moving average of gradients (momentum)
        - v: Moving average of squared gradients (velocity)
        - t: Timestep counter for bias correction
        
        These values are used in update_parameters() to compute adaptive learning rates
        for each parameter individually.
        """
        self.adam = {
            'weights': [{
                'm': np.zeros_like(w),  # First moment (momentum)
                'v': np.zeros_like(w),  # Second moment (RMSprop)
            } for w in self.weights],
            'bias': [{
                'm': np.zeros_like(b),
                'v': np.zeros_like(b),
            } for b in self.bias],
            't': 0  # Timestep
        }
    
    def update_parameters(self, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Updates network parameters using the Adam optimization algorithm.
        Adam (Adaptive Moment Estimation) maintains per-parameter learning rates
        adapted based on estimates of first and second moments of the gradients.
        
        Args:
            grads: Dictionary containing gradients for weights and biases
            learning_rate: Step size for parameter updates (default: 0.001)
            beta1: Exponential decay rate for moment estimates (default: 0.9)
            beta2: Exponential decay rate for squared moment estimates (default: 0.999)
            epsilon: Small constant to prevent division by zero (default: 1e-8)
        
        The Adam update rule for each parameter θ:
        1. Update biased first moment: m = β1*m + (1-β1)*g
        2. Update biased second moment: v = β2*v + (1-β2)*g²
        3. Apply bias correction: m_hat = m/(1-β1^t), v_hat = v/(1-β2^t)
        4. Update parameters: θ = θ - α * m_hat/(√v_hat + ε)
        
        Where:
        - g is the current gradient
        - t is the timestep
        - α is the learning rate
        """
        # Initialize Adam parameters if not already done
        if not hasattr(self, 'adam'):
            self.init_adam()
        
        # Increment timestep
        self.adam['t'] += 1
        t = self.adam['t']
        
        # Bias correction terms
        m_corr = 1 / (1 - beta1**t)
        v_corr = 1 / (1 - beta2**t)
        
        # Update each parameter (weights and biases)
        for param_type in ['weights', 'bias']:
            for l in range(len(self.weights)):
                # Get gradients and Adam parameters for this layer
                grad = grads[param_type][l]
                m = self.adam[param_type][l]['m']
                v = self.adam[param_type][l]['v']
                
                # Update biased first moment estimate
                m = beta1 * m + (1 - beta1) * grad
                # Update biased second raw moment estimate
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                
                # Store updated moments
                self.adam[param_type][l]['m'] = m
                self.adam[param_type][l]['v'] = v
                
                # Compute bias-corrected moments
                m_hat = m * m_corr
                v_hat = v * v_corr
                
                # Update parameters
                if param_type == 'weights':
                    self.weights[l] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                else:
                    self.bias[l] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def accuracy(self, predictions, targets):
        """
        Computes the accuracy of the network on a batch of inputs and targets.
        
        Args:
            predictions: Output probabilities from the network, shape (batch_size, num_classes)
                        For MNIST: (batch_size, 10) where each row contains probabilities for digits 0-9
            targets: Target labels in one-hot format, shape (batch_size, num_classes)
                    For MNIST: (batch_size, 10) where each row has a 1 at the true digit's position
        
        Returns:
            float: Accuracy of the network on the batch (between 0 and 1)
                  Computed as the fraction of correctly classified examples
        """
        predicted_classes = np.argmax(predictions, axis=1)  # Get the most likely class for each example
        true_classes = np.argmax(targets, axis=1)          # Get the true class from one-hot encoding
        return np.mean(predicted_classes == true_classes)   # Compute fraction of correct predictions

    def save_weights(self):
        """Save current weights and biases"""
        return [(w.copy(), b.copy()) for w, b in zip(self.weights, self.bias)]
    
    def load_weights(self, saved_weights):
        """Load saved weights and biases"""
        self.weights = [w.copy() for w, b in saved_weights]
        self.bias = [b.copy() for w, b in saved_weights]

    def train_step(self, inputs, targets, learning_rate=0.001):
        """
        Performs one complete training step: forward pass, loss computation,
        backpropagation, and parameter updates using Adam optimizer.
        
        Args:
            inputs: Input data batch, shape (batch_size, input_features)
                   For MNIST: (batch_size, 784) where each row is a flattened 28x28 image
            targets: Target labels in one-hot format, shape (batch_size, num_classes)
                    For MNIST: (batch_size, 10) where each row has a 1 at the true digit's position
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
        
        Returns:
            float: Cross-entropy loss value for this batch
        
        The training step follows this sequence:
        1. Forward pass: Compute predictions and cache intermediate values
        2. Loss computation: Calculate cross-entropy loss between predictions and targets
        3. Backward pass: Compute gradients for all parameters
        4. Parameter update: Apply Adam optimization to update weights and biases
        
        Note: The loss is clipped to prevent numerical instability from log(0)
        """
        # Forward pass
        predictions, cache = self.forward(inputs)
        
        # Compute cross-entropy loss
        epsilon = 1e-15  # Small number to prevent log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
        # Backward pass
        grads = self.backward(cache, targets)
        
        # Update parameters using Adam
        self.update_parameters(grads, learning_rate=learning_rate)

        acc = self.accuracy(predictions, targets)
        
        return loss, acc

    def fit(self, inputs, targets, epochs, batch_size=128, learning_rate=0.001, 
            patience=5, min_improvement=1e-4, verbose=True, use_best_weights=True):
        """
        Trains the neural network with the option to use best weights.
        
        Args:
            inputs: Training data, shape (n_samples, input_features)
            targets: Target labels in one-hot format, shape (n_samples, num_classes)
            epochs: Number of complete passes through the training data
            batch_size: Number of samples per gradient update (default: 128)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            patience: Number of epochs to wait for improvement before stopping (default: 5)
            min_improvement: Minimum improvement in loss to be considered significant (default: 1e-4)
            verbose: If True, prints training progress for each epoch (default: True)
            use_best_weights: If True, restore the weights from the best epoch after training (default: True)
        
        Returns:
            dict: Training history containing:
                - 'loss': List of training losses per epoch
                - 'accuracy': List of training accuracies per epoch
                - 'stopped_early': Boolean indicating if training stopped early
                - 'best_epoch': Epoch number with the best performance
        """
        # Convert inputs and targets to float32
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        n_samples = len(inputs)
        history = {
            'loss': [],
            'accuracy': [],
            'stopped_early': False,
            'best_epoch': 0
        }
        
        # Early stopping variables
        best_loss = float('inf')
        best_weights = None
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Shuffle the training data at the start of each epoch
            permutation = np.random.permutation(n_samples)
            inputs_shuffled = inputs[permutation]
            targets_shuffled = targets[permutation]
            
            # Initialize epoch metrics
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            
            # Train on mini-batches
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_inputs = inputs_shuffled[batch_start:batch_end]
                batch_targets = targets_shuffled[batch_start:batch_end]
                
                # Perform one training step
                batch_loss, batch_accuracy = self.train_step(
                    batch_inputs, 
                    batch_targets,
                    learning_rate
                )
                
                # Update epoch metrics
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                n_batches += 1
            
            # Compute average metrics for the epoch
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            
            # Store metrics in history
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            
            # Print progress if verbose is True
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"loss: {epoch_loss:.4f} - "
                      f"accuracy: {epoch_accuracy:.4f}")
            
            # Save best weights if we found a better model
            if epoch_loss < best_loss - min_improvement:
                # We found a better model
                best_loss = epoch_loss
                best_weights = self.save_weights()
                epochs_without_improvement = 0
                history['best_epoch'] = epoch
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping: No significant improvement for {patience} epochs")
                history['stopped_early'] = True
                break
        
        if not history['stopped_early'] and verbose:
            print("\nTraining completed for all epochs")
        
        # Restore best weights if requested
        if use_best_weights and best_weights is not None:
            if verbose:
                print(f"\nRestoring weights from best epoch ({history['best_epoch'] + 1})")
            self.load_weights(best_weights)
        elif verbose and use_best_weights:
            print("\nKeeping final weights (no better weights found)")
        
        return history

    def predict(self, inputs, return_probabilities=False):
        """
        Make predictions on input data.
        
        Args:
            inputs: Input data of shape (n_samples, input_features)
            return_probabilities: If True, returns probability distribution over classes
                                If False, returns predicted class indices (default: False)
        
        Returns:
            If return_probabilities is True:
                Probability distribution over classes, shape (n_samples, num_classes)
            If return_probabilities is False:
                Predicted class indices, shape (n_samples,)
        """
        # Ensure inputs are numpy array and correct dtype
        inputs = np.asarray(inputs, dtype=np.float32)
        
        # Forward pass through the network
        probabilities, _ = self.forward(inputs)
        
        if return_probabilities:
            return probabilities
        else:
            # Return predicted class (index of highest probability)
            return np.argmax(probabilities, axis=1)

    def save_model(self, filepath):
        """
        Save the neural network model to a file.
        
        This method saves the model architecture, weights, biases, and optimizer state.
        The file is saved in JSON format with numpy arrays converted to lists.
        
        Args:
            filepath (str): Path where the model should be saved
                          Example: 'models/mnist_model.json'
        """
        model_data = {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'weights': [w.tolist() for w in self.weights],
            'bias': [b.tolist() for b in self.bias],
            'adam': {
                'weights': [{
                    'm': w['m'].tolist(),
                    'v': w['v'].tolist()
                } for w in self.adam['weights']],
                'bias': [{
                    'm': b['m'].tolist(),
                    'v': b['v'].tolist()
                } for b in self.adam['bias']],
                't': self.adam['t']
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a neural network model from a file.
        
        This method creates a new NeuralNetwork instance with the saved architecture
        and loads the weights, biases, and optimizer state.
        
        Args:
            filepath (str): Path to the saved model file
                          Example: 'models/mnist_model.json'
        
        Returns:
            NeuralNetwork: A new instance of NeuralNetwork with the loaded model state
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            json.JSONDecodeError: If the model file is invalid
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Create new network with same architecture
        network = cls(model_data['input_size'], model_data['hidden_layers'], model_data['output_size'])
        
        # Load weights and biases
        network.weights = [np.array(w) for w in model_data['weights']]
        network.bias = [np.array(b) for b in model_data['bias']]
        
        # Load optimizer state
        network.adam = {
            'weights': [{
                'm': np.array(w['m']),
                'v': np.array(w['v'])
            } for w in model_data['adam']['weights']],
            'bias': [{
                'm': np.array(b['m']),
                'v': np.array(b['v'])
            } for b in model_data['adam']['bias']],
            't': model_data['adam']['t']
        }
        
        return network
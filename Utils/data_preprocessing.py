import numpy as np

def to_one_hot(y, num_classes=10):
    """
    Convert class labels to one-hot encoded format.
    
    Args:
        y: Array of class labels (integers), shape (n_samples,)
        num_classes: Number of classes (default: 10 for MNIST)
    
    Returns:
        Array of one-hot encoded labels, shape (n_samples, num_classes)
    
    Example:
        >>> labels = np.array([2, 1, 0, 3])
        >>> to_one_hot(labels, num_classes=4)
        array([[0., 0., 1., 0.],
               [0., 1., 0., 0.],
               [1., 0., 0., 0.],
               [0., 0., 0., 1.]])
    """
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

def preprocess_images(images, flatten=True):
    """
    Preprocess image data for neural network training.
    
    Args:
        images: Input images array of shape (n_samples, height, width) or (n_samples, height, width, channels)
               For MNIST: (n_samples, 28, 28) grayscale images
        flatten: If True, flattens 2D/3D images into 1D arrays (default: True)
    
    Returns:
        Preprocessed images with:
        - Pixel values normalized to [0, 1] range
        - Optionally flattened into 1D arrays
        
    Example:
        >>> x_train = preprocess_images(raw_images)
        >>> x_train.shape  # If raw_images was (60000, 28, 28), output will be (60000, 784)
    """
    # Convert to float32 for better numerical precision
    X = images.astype(np.float32)
    
    # Normalize pixel values to [0, 1] range
    X = X / 255.0
    
    # Flatten images if requested
    if flatten:
        X = X.reshape(X.shape[0], -1)
    
    return X

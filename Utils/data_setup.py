import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf

def get_data(type: str):
    """
    Load the MNIST dataset.
    
    Args:
        type (str): Type of data to be loaded.
            Must be 'train' for training data or 'test' for test data.
    
    Returns:
        tuple: A pair of arrays (images, labels) where:
            - images: Numpy array with MNIST images
            - labels: Numpy array with corresponding labels
    
    Raises:
        ValueError: If the type is not 'train' or 'test'
    """
    (train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.mnist.load_data()
    
    if type == 'train':
        return (train_imgs, train_lbls)
    elif type == 'test':
        return (test_imgs, test_lbls)
    else:
        raise ValueError("Invalid type. Must be 'train' or 'test'.")

if __name__ == '__main__':
    test_data = get_data('test')
    for i in range(2):
        print(test_data[0][i])
        print(test_data[1][i])
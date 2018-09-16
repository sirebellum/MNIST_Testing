import tensorflow as tf
import numpy as np
      
# Define the input function for training
def train_function(x,
               y=None,
               batch_size=128,
               num_epochs=None,
               shuffle=True,
               queue_capacity=51200,
               num_threads=6):

    return tf.estimator.inputs.numpy_input_fn(x,
                                              y=y,
                                              batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              shuffle=shuffle,
                                              queue_capacity=queue_capacity,
                                              num_threads=num_threads)

# Define the input function for evaluating
def eval_function(x,
                  y=None,
                  batch_size=128,
                  num_epochs=1,
                  shuffle=True,
                  queue_capacity=1000,
                  num_threads=1):

    return tf.estimator.inputs.numpy_input_fn(x,
                                              y=y,
                                              batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              shuffle=shuffle,
                                              queue_capacity=queue_capacity,
                                              num_threads=num_threads)
                        
# get weights from checkpoint at weights                        
def get_weights(weights):

    # Read ckpt
    checkpoint_path = weights
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Names of variables
    variables = [key for key in var_to_shape_map]
    # Only get important layers, marked with "-" at the ends
    variables = [var for var in variables if "-" in var]
    # Get actual layer names (instead of variable names)
    variables = set([var.split("-")[0] for var in variables])
    # Sort by sequence in model, first to last
    from natsort import natsorted, ns
    variables = natsorted(variables, key=lambda y: y.lower())
    
    # Collect various weights for layers
    conv_biases = [np.asarray( reader.get_tensor(key+"-/bias") ) \
                                for key in variables]
    conv_kernels = [np.asarray( reader.get_tensor(key+"-/kernel") ) \
                                for key in variables]
    
    # Aggregate variables
    variables = {}
    variables['conv_biases'] = conv_biases
    variables['conv_kernels'] = conv_kernels
        
    return variables
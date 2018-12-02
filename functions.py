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
    # Only get important layers
    variables = [var for var in variables if 'bias' in var or 'kernel' in var]
    # Get actual layer names (instead of variable names)
    variables = set([var.split("/")[0] for var in variables])
    # Sort by sequence in model, first to last
    from natsort import natsorted, ns
    variables = natsorted(variables, key=lambda y: y.lower())
    
    # Collect various weights for layers
    biases = [np.asarray( reader.get_tensor(key+"/bias") ) \
                                for key in variables]
    kernels = [np.asarray( reader.get_tensor(key+"/kernel") ) \
                                for key in variables]
    
    # Aggregate variables
    variables = {}
    variables['biases'] = biases
    variables['kernels'] = kernels
        
    return variables
    
if __name__ == '__main__':

    saved_model = "models/final/model.ckpt-300000"
    model_data = get_weights(saved_model)
    
    # Get max value in all layers
    maxes = list()
    for kernel in model_data['kernels']:
        maxes.append(np.max(np.absolute(kernel)))
    for bias in model_data['biases']:
        maxes.append(np.max(np.absolute(bias)))
    max_value = max(maxes)
    
    # Quantize
    kernels = list()
    for l in model_data['kernels']:
    
        # Each kernel in layer
        kernel = list()
        for output in l:
        
            # Each output in kernel
            conn = list()
            for val in output:
                v = str(hex(int(val/max_value * 4096)))
                conn.append( v )
            kernel.append(conn)
            
        # Make neuron-wise
        kernel = list(zip(*kernel))
            
        kernels.append(kernel)
    
    biases = list()
    for l in model_data['biases']:
    
        # Each bias in layer
        bias = list()
        for val in l:
        
            v = str(hex(int(val/max_value * 4096)))
            bias.append( v )
            
        biases.append(bias)
    
    with open("model.dat", "w") as f: 
    
        line = "{} "*(len(kernels[0][0])+2)
        for x in range(0, len(kernels)):
            
            filler = [0]*(len(kernels[0][0])-len(kernels[x][0]))
            
            for n in range(0, len(kernels[x])):
                f.write(line.format(x, *kernels[x][n], *filler, biases[x][n])+"\n")
            
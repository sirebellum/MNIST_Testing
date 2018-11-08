import tensorflow as tf
HEIGHT = 28
WIDTH = 28

def encode(features, weights):
  
  # Hidden Layers
  flattened = tf.reshape(features, [-1, HEIGHT*WIDTH])
  hidden_layer1 = tf.layers.dense(inputs=flattened, units=256, activation=tf.nn.relu)
  hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=64, activation=tf.nn.relu)
  hidden_layer3 = tf.layers.dense(inputs=hidden_layer2, units=32, activation=tf.nn.relu)
  
  return hidden_layer3
  
def autoencode(features, labels, mode, params):
  
    # Input Layer
    print("Mode:", mode)
    input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, 1], name="image_input")
    
    # Parameters setup
    weights = params['weights']
    
    # Encode stuff
    feature_map = encode(input_layer, weights)

    # Print dimensionality of feature map
    _, length = feature_map.get_shape()
    print("CNN with final features:", length)

    # Decoding Layer
    reconstructed = tf.layers.dense(inputs=feature_map, units=HEIGHT*WIDTH, activation=tf.nn.relu)

    # Reshape to image
    reconstructed = tf.reshape(reconstructed, [-1, WIDTH, HEIGHT, 1], name="image_output")

    # Calculate Loss
    loss = tf.losses.mean_squared_error(labels=input_layer,
                                        predictions=reconstructed)
                                   
    # Put images in tensorboard
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.image(
            "Image",
            reconstructed,
            max_outputs=18
        )                
                
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(epsilon=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    # EVAL stuff
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "RMSE": tf.metrics.root_mean_squared_error(
          labels=input_layer, predictions=reconstructed)
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
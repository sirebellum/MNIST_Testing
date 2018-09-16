import tensorflow as tf

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.5, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def encode(features, labels, mode, params):

    # Input Layer
    print("Mode:", mode)
    input_layer = tf.reshape(features, [-1, 28, 28, 1], name="image_input")
    
    # Add noise
    noisy_layer = gaussian_noise_layer(input_layer, 1)

    # Encoder
    conv1_1 = tf.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(noisy_layer)
    pool1 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(conv1_1)
    conv1_2 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(conv1_2)
    conv1_3 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
    h = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='feature_map')(conv1_3)

    # Print dimensionality of lowest level
    _, height, width, depth = h.get_shape()
    print("CNN with final feature maps:", height, "x", width, "x", depth)
    print(height*width*depth, "total features")
    
    # Decoder
    conv2_1 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(h)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv2_1)
    conv2_2 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D((2, 2))(conv2_2)
    conv2_3 = tf.layers.Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = tf.image.resize_nearest_neighbor(conv2_3, [28, 28]) # Rplaced UpSamplig2D b/c bug
    reconstructed = tf.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', 
                                     activity_regularizer=tf.nn.l2_loss,
                                     name='reconstructed_image')(up3)

    # Calculate Loss
    loss = tf.losses.mean_squared_error(labels=input_layer,
                                      predictions=reconstructed)
                                   
    # Put images in tensorboard
    tf.summary.image(
        "noisy",
        noisy_layer,
        max_outputs=9
      )
    tf.summary.image(
        "reconstructed",
        reconstructed,
        max_outputs=9
      )
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # reducing e to 0.0001 midway through training increases accuracy
        optimizer = tf.train.AdamOptimizer(epsilon=0.0001)
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

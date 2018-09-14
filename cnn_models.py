import tensorflow as tf
NUMCLASSES = 10

def CNN_Model(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  print("Mode:", mode)
  input_layer = tf.reshape(features, [-1, 28, 28, 1], name="image_input")

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[5, 5],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu)
  
  # Pool Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=(1,1))
  
  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu)
      
  # Pool Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=(1,1))
  
  # Final layer for conversion
  final = pool2
      
  # Dense Layer
  _, height, width, depth = final.get_shape()
  print("CNN with final feature maps:", height, "x", width, "x", depth)
  print(height*width*depth, "total features")
  final_flat = tf.reshape(final, [-1, height * width * depth])
  dense = tf.layers.dense(inputs=final_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=NUMCLASSES)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  #Put images in tensorboard
  if mode == tf.estimator.ModeKeys.TRAIN:
      tf.summary.image(
        "Image",
        input_layer,
        max_outputs=9
      )
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "mean_accuracy": tf.metrics.mean_per_class_accuracy(
          labels=labels, predictions=predictions["classes"],
          num_classes=NUMCLASSES)
  }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
      
      
def parse_record(serialized_example): #parse a single binary example
  """Parses a single tf.Example into image and label tensors."""
  features = {'image/encoded': tf.FixedLenFeature([], tf.string),
             'image/format':  tf.FixedLenFeature([], tf.string),
             'image/label':   tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(serialized_example, features)
  
  #print("JPG:", features['image/encoded'])
  image = tf.image.decode_jpeg(features['image/encoded'], channels=0)
  #print("image:", image)
  image = tf.reshape(image, [40, 398, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  
  label = tf.cast(features['image/label'], tf.int32)
  
  return (image, label)
  
# Define the input function for training
def train_function(x,
               y=None,
               batch_size=128,
               num_epochs=None,
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

# Define the input function for evaluating
def eval_function(x,
                  y=None,
                  batch_size=128,
                  num_epochs=70,
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
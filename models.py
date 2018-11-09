import tensorflow as tf
HEIGHT = 28
WIDTH = 28
BETA = 0.001

def classifier(features, labels, mode, params):

  # Input Layer
  print("Mode:", mode)
  input_layer = tf.reshape(features, [-1, 28, 28, 1], name="image_input")

  # For classification purposes
  NUMCLASSES = params['num_labels']

  # Feature extractor (function)
  extract = params['feature_extractor']
  
  # Pretrained weights
  weights = params['weights']
  
  # Extract final layer for classification
  feature_map = extract(input_layer, weights)

  # Final feature map dimensions
  #_, height, width, depth = feature_map.get_shape()
  #print("CNN with final feature maps:", height, "x", width, "x", depth)
  #print(height*width*depth, "total features")
  
  # Dense layer
  #final_flat = tf.reshape(feature_map, [-1, HEIGHT*WIDTH])
  dropout = tf.layers.dropout(
      inputs=feature_map, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
                           units=NUMCLASSES)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  # Put images in tensorboard
  if mode == tf.estimator.ModeKeys.TRAIN:
      input_layer = tf.reshape(features, [-1, 28, 28, 1], name="image_input")
      tf.summary.image(
        "Image",
        input_layer,
        max_outputs=9
      )
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # L2 Regularization for logits
  loss += tf.reduce_mean(tf.losses.get_regularization_losses())

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(epsilon=0.0001)
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
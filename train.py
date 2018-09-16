# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from functions import classifier, train_function, get_weights
import mnist

# Autoencoders
from autoencoders import conv, vanilla

# which encoder to use
feature_extractor = conv.encode

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
parser.add_argument("--steps", default=5000, help="Train to number of steps")
parser.add_argument("--weights", default=None, help="Model checkpoint to get pretrained weights from")
args = parser.parse_args()
num_steps = int(args.steps)

# Directory setup
abs_path = os.path.abspath(__file__) # Absolute path of this file
directory = os.path.dirname(abs_path)
model_dir = directory+"/models/"+args.output_name

# Get pretrained weights for feature extractor
weights = None
if args.weights is not None:
    weights = os.path.join(os.path.dirname(__file__),'models', args.weights)
    weights = get_weights(weights)

def main(unused_argv):

  # Set up 
  train_input_fn = train_function(mnist.train_images, y=mnist.train_labels)
  
  # Define params for model
  params = {}
  params['num_labels'] = len( set(mnist.train_labels) )
  params['feature_extractor'] = feature_extractor
  params['noise'] = True

  # Estimator config to change frequency of ckpt files
  estimator_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 10,  # Save checkpoints every 10 seconds
    keep_checkpoint_max = 2)       # Retain the 2 most recent checkpoints.
  
  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=conv.autoencoder,
    model_dir=model_dir,
    config=estimator_config,
    params=params)
    
  # Set up logging for predictions
  #tensors_to_log = {"predictions": "image_output"}
  #logging_hook = tf.train.LoggingTensorHook(
  #    tensors=tensors_to_log, every_n_iter=50)
      
  # Train the model
  classifier.train(
        input_fn=train_input_fn,
        steps=num_steps)

if __name__ == "__main__":
  tf.app.run()

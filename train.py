# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_models import CNN_Model, train_function
import mnist

# Autoencoders
from autoencoders import conv, vanilla, convnoise

# which model to use
cnn_model = convnoise.encode

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
parser.add_argument("--steps", default=5000, help="Train to number of steps")
args = parser.parse_args()
num_steps = int(args.steps)

CWD_PATH = os.getcwd()

def main(unused_argv):

  # Set up 
  train_input_fn = train_function(mnist.train_images, y=mnist.train_labels)
  
  # Define params for model
  params = {}
  params['num_labels'] = len( set(mnist.train_labels) )

  # Estimator config to change frequency of ckpt files
  estimator_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 10,  # Save checkpoints every 10 seconds
    keep_checkpoint_max = 2)       # Retain the 2 most recent checkpoints.
  
  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=CWD_PATH+"/models/"+args.output_name,
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

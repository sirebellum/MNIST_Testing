# Imports
import numpy as np
import os
import argparse
import inotify.adapters
import tensorflow as tf
from functions import classifier, eval_function, get_weights
import mnist

# Autoencoders
from autoencoders import conv, vanilla

# Which model to use
feature_extractor = conv.encode

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Relative path to model")
parser.add_argument("--eval", default=0, help="Evaluate only the most recent checkpoint if set")
parser.add_argument("--weights", default=None, help="Model checkpoint to get pretrained weights from")
args = parser.parse_args()

# Directory setup
abs_path = os.path.abspath(__file__) # Absolute path of this file
directory = os.path.dirname(abs_path)
model_dir = directory+"/models/"+args.output_name

# Get pretrained weights for feature extractor
weights = None
if args.weights is not None:
    weights = os.path.join(os.path.dirname(__file__),'models', args.weights)
    weights = get_weights(weights)

#Inotify setup
file_watch = inotify.adapters.Inotify()
file_watch.add_watch(model_dir)

def main(unused_argv):

  # Get data
  eval_input_fn = eval_function(mnist.test_images, y=mnist.test_labels)
  
  # Define params for model
  params = {}
  params['num_labels'] = len( set(mnist.train_labels) )
  params['feature_extractor'] = feature_extractor

  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=conv.autoencoder,
    model_dir=model_dir,
    params=params)

  # Evaluate immediately
  print("Evaluating...")
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  if args.eval: # exit if flag set
    exit() 
  
  # Evaluate for every new file
  for event in file_watch.event_gen(yield_nones=False):
    # Evaluate the model and print results
    (_, type_names, path, filename) = event
    new_ckpt = type_names[0] is 'IN_MOVED_TO' and 'checkpoint' in filename and 'tmp' not in filename
    if new_ckpt:
      print("Evaluating...")
      eval_results = classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
  
if __name__ == "__main__":
  tf.app.run()

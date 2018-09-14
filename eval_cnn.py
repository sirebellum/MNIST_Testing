# Imports
import numpy as np
import os
import argparse
import inotify.adapters
import tensorflow as tf
from cnn_models import CNN_Model, parse_record, eval_function
cnn_model = CNN_Model #which model to use
import mnist

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Relative path to model")
parser.add_argument("--eval", default=0, help="Evaluate only the most recent checkpoint if set")
args = parser.parse_args()

#Allow either model name or directory to be used
CWD_PATH = os.getcwd()
if "models" and "/" not in args.output_name:
  model_path = CWD_PATH+"/models/"+args.output_name
else:
  model_path = CWD_PATH+"/"+args.output_name
print("Set to evaluate model at", model_path)

#Inotify setup
file_watch = inotify.adapters.Inotify()
file_watch.add_watch(model_path)

def main(unused_argv):

  # Get data
  eval_input_fn = eval_function(mnist.test_images, y=mnist.test_labels)

  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=model_path)

  if args.eval:
    print("Evaluating...")
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    return 0
    
  for event in file_watch.event_gen(yield_nones=False): #Evaluate for every new file
    # Evaluate the model and print results
    (_, type_names, path, filename) = event
    new_ckpt = type_names[0] is 'IN_MOVED_TO' and 'checkpoint' in filename and 'tmp' not in filename
    if new_ckpt:
      print("Evaluating...")
      eval_results = classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
  
if __name__ == "__main__":
  tf.app.run()

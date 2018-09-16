import csv
import numpy as np

test_labels = list()
test_images = list()
with open('mnist_test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        test_labels.append(row[0])
        image = [ row[0+x:28+x] for x in range(0, 28*28, 28) ]
        test_images.append(image)
        
train_labels = list()
train_images = list()
with open('mnist_train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        train_labels.append(row[0])
        image = [ row[0+x:28+x] for x in range(0, 28*28, 28) ]
        train_images.append(image)
        
# Convert to numpy arrays
test_labels = np.asarray(test_labels, dtype=np.dtype('int32'))
test_images = np.asarray(test_images, dtype=np.dtype('float32'))
train_labels = np.asarray(train_labels, dtype=np.dtype('int32'))
train_images = np.asarray(train_images, dtype=np.dtype('float32'))
# Convert to float
test_images = np.divide(test_images, 255)
train_images = np.divide(train_images, 255)

#import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    from PIL import Image
    for data in test_images:
        img = Image.fromarray(data, 'L')
        img.show()
        input()
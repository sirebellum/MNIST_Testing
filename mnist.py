import csv
import numpy as np

test_labels = list()
test_images = list()
with open('mnist_test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        test_labels.append(row[0])
        image = [ row[0+x:27+x] for x in range(0, 28*28, 28) ]
        test_images.append(image)
        
train_labels = list()
train_images = list()
with open('mnist_train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        train_labels.append(row[0])
        image = [ row[0+x:27+x] for x in range(0, 28*28, 28) ]
        train_images.append(image)
        
# COnvert to numpy arrays
test_labels = np.asarray(test_labels, dtype=np.uint8)
test_images = np.asarray(test_images, dtype=np.uint8)
train_labels = np.asarray(train_labels, dtype=np.uint8)
train_images = np.asarray(train_images, dtype=np.uint8)

#import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    from PIL import Image
    for instance in test_images:
        img = Image.fromarray(instance, 'L')
        img.show()
        input()
        img.close()
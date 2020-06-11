# Intervertebral Disc Segmentation
## Image Dataset Loader (2D)
## Intervertebral Disc Segmentation

import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


input_image_root = './Dataset/Disc_png/Dataset'
label_image_root = './Dataset/Disc_png/Label'

# Dataset counting
print(len(os.listdir(input_image_root)))
print(len(os.listdir(label_image_root)))

# Get absolute path of input data
input_file_path_list = []
input_path_list = [os.path.join(input_image_root, dir) for dir in os.listdir(input_image_root)]
for path in input_path_list:
    for file in os.listdir(path):
        input_file_path_list.append(os.path.join(path, file))

# Dataset random shuffle
np.random.shuffle(input_file_path_list)

# Get absolute path of label data
label_file_path_list = [path.replace('/Disc_png/Dataset', '/Disc_png/Label') for path in input_file_path_list]

# Check file path list of absolute path(input data)
print(len(input_file_path_list))
print(input_file_path_list[0])
print(os.path.exists(input_file_path_list[0]))

# Check file path list of absolute path(label data)
print(len(label_file_path_list))
print(label_file_path_list[0])
print(os.path.exists(label_file_path_list[0]))

# Split dataset for train(9) : test(1)
split_n = int(round(len(input_file_path_list) / 10))
train_input_split = np.array(input_file_path_list[split_n:])
test_input_split = np.array(input_file_path_list[:split_n])
train_label_split = np.array(label_file_path_list[split_n:])
test_label_split = np.array(label_file_path_list[:split_n])
print('Train input count : {}, Train label count : {}'.format(len(train_input_split), len(train_label_split)))
print('Test input count : {}, Test label count : {}'.format(len(test_input_split), len(test_label_split)))

#train_input_split = train_input_split[:1000]
#train_label_split = train_label_split[:1000]


train_input_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in train_input_split]
train_label_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in train_label_split]
test_input_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in test_input_split]
test_label_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in test_label_split]

train_input_list, train_label_list = shuffle(train_input_list, train_label_list)

train_input = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in train_input_list]), axis=-1)
train_label = np.transpose(np.array([[cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY_INV)[1], cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY)[1]] for img in train_label_list]), axes=[0, 2, 3, 1]) / 255
print(train_input.shape, train_label.shape)

validation_input = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in test_input_list]), axis=-1)
validation_label = np.transpose(np.array([[cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY_INV)[1], cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY)[1]] for img in test_label_list]), axes=[0, 2, 3, 1]) / 255
print(validation_input.shape, validation_label.shape)


np.savez_compressed('./Dataset/trainset_256.npz', dataset=train_input, label=train_label)
np.savez_compressed('./Dataset/testset_256.npz', dataset=validation_input, label=validation_label)

#train_loaded = np.load('./Dataset/trainset.npz')
#test_loaded = np.load('./Dataset/testset.npz')

#train_input = train_loaded['dataset']
#train_label = train_loaded['label']

#validation_input = test_loaded['dataset']
#validation_label = test_loaded['label']


## Intervertebral Disc Segmentation
import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ## IF GPU RUN, ELIMINATE THIS CODE
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

input_image_root = 'E:/PythonProjects/HW/DL_projects/Dataset/Disc_png/Dataset'
label_image_root = 'E:/PythonProjects/HW/DL_projects/Dataset/Disc_png/Label'

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

# %%

## Segmentation performance analysis function
from tensorflow.keras import backend as K


# IoU
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


# Dice score
def dice_coef(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# %%

## Model Build

from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Concatenate


# Unet Network Architecture
def Unet(input_shape=[256, 256, 1], channel_size=8):
    n1_Input_o1 = Input(shape=input_shape, batch_shape=None, name='n1_Input')
    n2_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n1_Input_o1)
    n2_contract_n2_BatchNormalization_o1 = BatchNormalization()(n2_contract_n1_Conv2D_o1)
    n2_contract_n3_Activation_o1 = Activation(activation='relu')(n2_contract_n2_BatchNormalization_o1)
    n2_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n2_contract_n3_Activation_o1)
    n2_contract_n5_BatchNormalization_o1 = BatchNormalization()(n2_contract_n4_Conv2D_o1)
    n2_contract_n6_Activation_o1 = Activation(activation='relu')(n2_contract_n5_BatchNormalization_o1)
    n5_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n2_contract_n6_Activation_o1)
    n6_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n5_MaxPooling2D_o1)
    n6_contract_n2_BatchNormalization_o1 = BatchNormalization()(n6_contract_n1_Conv2D_o1)
    n6_contract_n3_Activation_o1 = Activation(activation='relu')(n6_contract_n2_BatchNormalization_o1)
    n6_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n6_contract_n3_Activation_o1)
    n6_contract_n5_BatchNormalization_o1 = BatchNormalization()(n6_contract_n4_Conv2D_o1)
    n6_contract_n6_Activation_o1 = Activation(activation='relu')(n6_contract_n5_BatchNormalization_o1)
    n7_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n6_contract_n6_Activation_o1)
    n8_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n7_MaxPooling2D_o1)
    n8_contract_n2_BatchNormalization_o1 = BatchNormalization()(n8_contract_n1_Conv2D_o1)
    n8_contract_n3_Activation_o1 = Activation(activation='relu')(n8_contract_n2_BatchNormalization_o1)
    n8_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n8_contract_n3_Activation_o1)
    n8_contract_n5_BatchNormalization_o1 = BatchNormalization()(n8_contract_n4_Conv2D_o1)
    n8_contract_n6_Activation_o1 = Activation(activation='relu')(n8_contract_n5_BatchNormalization_o1)
    n9_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n8_contract_n6_Activation_o1)
    n10_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n9_MaxPooling2D_o1)
    n10_contract_n2_BatchNormalization_o1 = BatchNormalization()(n10_contract_n1_Conv2D_o1)
    n10_contract_n3_Activation_o1 = Activation(activation='relu')(n10_contract_n2_BatchNormalization_o1)
    n10_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n10_contract_n3_Activation_o1)
    n10_contract_n5_BatchNormalization_o1 = BatchNormalization()(n10_contract_n4_Conv2D_o1)
    n10_contract_n6_Activation_o1 = Activation(activation='relu')(n10_contract_n5_BatchNormalization_o1)
    n11_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n10_contract_n6_Activation_o1)
    n3_bottle_neck_n1_Conv2D_o1 = Conv2D(filters=channel_size*16, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n11_MaxPooling2D_o1)
    n3_bottle_neck_n2_BatchNormalization_o1 = BatchNormalization()(n3_bottle_neck_n1_Conv2D_o1)
    n3_bottle_neck_n3_Activation_o1 = Activation(activation='relu')(n3_bottle_neck_n2_BatchNormalization_o1)
    n3_bottle_neck_n4_Conv2D_o1 = Conv2D(filters=channel_size*16, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n3_bottle_neck_n3_Activation_o1)
    n3_bottle_neck_n5_BatchNormalization_o1 = BatchNormalization()(n3_bottle_neck_n4_Conv2D_o1)
    n3_bottle_neck_n6_Activation_o1 = Activation(activation='relu')(n3_bottle_neck_n5_BatchNormalization_o1)
    n12_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size*8, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(n3_bottle_neck_n6_Activation_o1)
    n4_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n12_Conv2DTranspose_o1, n10_contract_n6_Activation_o1])
    n4_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n4_expand_n1_Concatenate_o1)
    n4_expand_n3_BatchNormalization_o1 = BatchNormalization()(n4_expand_n2_Conv2D_o1)
    n4_expand_n4_Activation_o1 = Activation(activation='relu')(n4_expand_n3_BatchNormalization_o1)
    n4_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n4_expand_n4_Activation_o1)
    n4_expand_n6_BatchNormalization_o1 = BatchNormalization()(n4_expand_n5_Conv2D_o1)
    n4_expand_n7_Activation_o1 = Activation(activation='relu')(n4_expand_n6_BatchNormalization_o1)
    n13_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size*4, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(n4_expand_n7_Activation_o1)
    n14_expand_n1_Concatenate_o1 = Concatenate(axis=-1, name='n14_expand_n1_Concatenate')([n13_Conv2DTranspose_o1, n8_contract_n6_Activation_o1])
    n14_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n14_expand_n1_Concatenate_o1)
    n14_expand_n3_BatchNormalization_o1 = BatchNormalization()(n14_expand_n2_Conv2D_o1)
    n14_expand_n4_Activation_o1 = Activation(activation='relu')(n14_expand_n3_BatchNormalization_o1)
    n14_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n14_expand_n4_Activation_o1)
    n14_expand_n6_BatchNormalization_o1 = BatchNormalization()(n14_expand_n5_Conv2D_o1)
    n14_expand_n7_Activation_o1 = Activation(activation='relu')(n14_expand_n6_BatchNormalization_o1)
    n15_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size*2, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(n14_expand_n7_Activation_o1)
    n16_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n15_Conv2DTranspose_o1, n6_contract_n6_Activation_o1])
    n16_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n16_expand_n1_Concatenate_o1)
    n16_expand_n3_BatchNormalization_o1 = BatchNormalization()(n16_expand_n2_Conv2D_o1)
    n16_expand_n4_Activation_o1 = Activation(activation='relu')(n16_expand_n3_BatchNormalization_o1)
    n16_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n16_expand_n4_Activation_o1)
    n16_expand_n6_BatchNormalization_o1 = BatchNormalization()(n16_expand_n5_Conv2D_o1)
    n16_expand_n7_Activation_o1 = Activation(activation='relu')(n16_expand_n6_BatchNormalization_o1)
    n17_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(n16_expand_n7_Activation_o1)
    n18_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n17_Conv2DTranspose_o1, n2_contract_n6_Activation_o1])
    n18_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n18_expand_n1_Concatenate_o1)
    n18_expand_n3_BatchNormalization_o1 = BatchNormalization()(n18_expand_n2_Conv2D_o1)
    n18_expand_n4_Activation_o1 = Activation(activation='relu')(n18_expand_n3_BatchNormalization_o1)
    n18_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(n18_expand_n4_Activation_o1)
    n18_expand_n6_BatchNormalization_o1 = BatchNormalization()(n18_expand_n5_Conv2D_o1)
    n18_expand_n7_Activation_o1 = Activation(activation='relu')(n18_expand_n6_BatchNormalization_o1)
    n19_Conv2D_o1 = Conv2D(filters=2, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='softmax', kernel_initializer='he_normal')(n18_expand_n7_Activation_o1)

    model = Model(inputs=n1_Input_o1, outputs=[n19_Conv2D_o1])
    return model


# %%

# Data split for 5 Fold Cross Validation
k = 3
now_fold = 0
kf = KFold(n_splits=k)
KFold(n_splits=k, random_state=None, shuffle=False)

## for debugging
#train_input_split = train_input_split[:1000]
#train_label_split = train_label_split[:1000]

result_dict = {}
for train_index, validation_index in kf.split(train_input_split):
    now_fold += 1
    print('Fold :', now_fold)
    # data random shuffle for regularization
    np.random.shuffle(train_index)

    # data split
    fold_train_input_list = train_input_split[train_index]
    fold_train_label_list = train_label_split[train_index]
    fold_validation_input_list = train_input_split[validation_index]
    fold_validation_label_list = train_input_split[validation_index]

    # image data read by using opencv
    fold_train_input_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_train_input_list]
    fold_train_label_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_train_label_list]
    fold_validation_input_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_validation_input_list]
    fold_validation_label_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_validation_label_list]

    # image data resizing by using opencv
    fold_train_input = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_train_input_list]), axis=-1)
    fold_train_label = np.transpose(np.array([[cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY_INV)[1], cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY)[1]] for img in fold_train_label_list]), axes=[0, 2, 3, 1]) / 255
    print(fold_train_input.shape, fold_train_label.shape)
    fold_validation_input = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_validation_input_list]),axis=-1)
    fold_validation_label = np.transpose(np.array([[cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY_INV)[1], cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY)[1]] for img in fold_validation_label_list]), axes=[0, 2, 3, 1]) / 255
    print(fold_validation_input.shape, fold_validation_label.shape)

    # Train Each Fold
    model = Unet(input_shape=[256, 256, 1], channel_size=16)
    model.summary()
    model.compile(optimizer=Nadam(lr=1e-5), loss=dice_coef_loss, metrics=[iou_coef, dice_coef])

    history = model.fit(fold_train_input,
                        fold_train_label,
                        #class_weight=[0.6, 1],
                        validation_data=(fold_validation_input, fold_validation_label),
                        epochs=100,
                        batch_size=16)

# %%



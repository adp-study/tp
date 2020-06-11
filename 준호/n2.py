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

input_image_root = './Dataset/Disc_2D_resampled_nonthickness_dilate_closing/Dataset'
label_image_root = './Dataset/Disc_2D_resampled_nonthickness_dilate_closing/Label'

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
label_file_path_list = [path.replace('/Disc_2D_resampled_nonthickness_dilate_closing/Dataset',
                                     '/Disc_2D_resampled_nonthickness_dilate_closing/Label') for path in
                        input_file_path_list]

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
# def dice_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
#     union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
#     dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
#     return dice
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# %%

## Model Build

from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Add, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, \
    UpSampling2D, concatenate
from tensorflow.keras import callbacks


# Unet Network Architecture
def UNet(input_size = (256,256,1), channel_size=16):
    inp = Input(input_size)
    conv1 = Conv2D(channel_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = Conv2D(channel_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(channel_size*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(channel_size*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(channel_size*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(channel_size*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(channel_size*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(channel_size*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(channel_size*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(channel_size*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(channel_size*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(channel_size*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(channel_size*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2D(channel_size*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(channel_size*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(channel_size*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2D(channel_size*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(channel_size*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(channel_size*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(channel_size, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(channel_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(channel_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inp, outputs=[conv10])

    return model

def Unet(input_shape=[256, 256, 1], channel_size=16):
    n1_Input_o1 = Input(shape=input_shape, batch_shape=None, name='n1_Input')
    n2_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], strides=[1, 1],
                                      padding='same', data_format=None, dilation_rate=[1, 1],
                                      activation=None, use_bias=True,
                                      kernel_initializer='he_normal', bias_initializer='Zeros',
                                      kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None, name='n2_contract_n1_Conv2D')(n1_Input_o1)
    n2_contract_n2_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                              epsilon=0.001, center=True,
                                                              scale=True,
                                                              beta_initializer='zeros',
                                                              gamma_initializer='ones',
                                                              moving_mean_initializer='zeros',
                                                              moving_variance_initializer='ones',
                                                              beta_regularizer=None,
                                                              gamma_regularizer=None,
                                                              beta_constraint=None,
                                                              gamma_constraint=None,
                                                              name='n2_contract_n2_BatchNormalization')(n2_contract_n1_Conv2D_o1)
    n2_contract_n3_Activation_o1 = Activation(activation='relu',name='n2_contract_n3_Activation')(n2_contract_n2_BatchNormalization_o1)
    n2_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], strides=[1, 1],
                                      padding='same', data_format=None, dilation_rate=[1, 1],
                                      activation=None, use_bias=True,
                                      kernel_initializer='he_normal', bias_initializer='Zeros',
                                      kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None, name='n2_contract_n4_Conv2D')(n2_contract_n3_Activation_o1)
    n2_contract_n5_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                              epsilon=0.001, center=True,
                                                              scale=True,
                                                              beta_initializer='zeros',
                                                              gamma_initializer='ones',
                                                              moving_mean_initializer='zeros',
                                                              moving_variance_initializer='ones',
                                                              beta_regularizer=None,
                                                              gamma_regularizer=None,
                                                              beta_constraint=None,
                                                              gamma_constraint=None,
                                                              name='n2_contract_n5_BatchNormalization')(n2_contract_n4_Conv2D_o1)
    n2_contract_n6_Activation_o1 = Activation(activation='relu', name='n2_contract_n6_Activation')(n2_contract_n5_BatchNormalization_o1)
    n5_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                      data_format=None, name='n5_MaxPooling2D')(n2_contract_n6_Activation_o1)
    n6_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], strides=[1, 1],
                                      padding='same', data_format=None, dilation_rate=[1, 1],
                                      activation=None, use_bias=True,
                                      kernel_initializer='he_normal', bias_initializer='Zeros',
                                      kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None, name='n6_contract_n1_Conv2D')(n5_MaxPooling2D_o1)
    n6_contract_n2_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                              epsilon=0.001, center=True,
                                                              scale=True,
                                                              beta_initializer='zeros',
                                                              gamma_initializer='ones',
                                                              moving_mean_initializer='zeros',
                                                              moving_variance_initializer='ones',
                                                              beta_regularizer=None,
                                                              gamma_regularizer=None,
                                                              beta_constraint=None,
                                                              gamma_constraint=None,
                                                              name='n6_contract_n2_BatchNormalization')(n6_contract_n1_Conv2D_o1)
    n6_contract_n3_Activation_o1 = Activation(activation='relu',name='n6_contract_n3_Activation')(n6_contract_n2_BatchNormalization_o1)
    n6_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], strides=[1, 1],
                                      padding='same', data_format=None, dilation_rate=[1, 1],
                                      activation=None, use_bias=True,
                                      kernel_initializer='he_normal', bias_initializer='Zeros',
                                      kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None, name='n6_contract_n4_Conv2D')(n6_contract_n3_Activation_o1)
    n6_contract_n5_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                              epsilon=0.001, center=True,
                                                              scale=True,
                                                              beta_initializer='zeros',
                                                              gamma_initializer='ones',
                                                              moving_mean_initializer='zeros',
                                                              moving_variance_initializer='ones',
                                                              beta_regularizer=None,
                                                              gamma_regularizer=None,
                                                              beta_constraint=None,
                                                              gamma_constraint=None,
                                                              name='n6_contract_n5_BatchNormalization')(n6_contract_n4_Conv2D_o1)
    n6_contract_n6_Activation_o1 = Activation(activation='relu', name='n6_contract_n6_Activation')(n6_contract_n5_BatchNormalization_o1)
    n7_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                      data_format=None, name='n7_MaxPooling2D')(n6_contract_n6_Activation_o1)
    n8_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], strides=[1, 1],
                                      padding='same', data_format=None, dilation_rate=[1, 1],
                                      activation=None, use_bias=True,
                                      kernel_initializer='he_normal', bias_initializer='Zeros',
                                      kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None, name='n8_contract_n1_Conv2D')(n7_MaxPooling2D_o1)
    n8_contract_n2_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                              epsilon=0.001, center=True,
                                                              scale=True,
                                                              beta_initializer='zeros',
                                                              gamma_initializer='ones',
                                                              moving_mean_initializer='zeros',
                                                              moving_variance_initializer='ones',
                                                              beta_regularizer=None,
                                                              gamma_regularizer=None,
                                                              beta_constraint=None,
                                                              gamma_constraint=None,
                                                              name='n8_contract_n2_BatchNormalization')(n8_contract_n1_Conv2D_o1)
    n8_contract_n3_Activation_o1 = Activation(activation='relu', name='n8_contract_n3_Activation')(n8_contract_n2_BatchNormalization_o1)
    n8_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], strides=[1, 1],
                                      padding='same', data_format=None, dilation_rate=[1, 1],
                                      activation=None, use_bias=True,
                                      kernel_initializer='he_normal', bias_initializer='Zeros',
                                      kernel_regularizer=None, bias_regularizer=None,
                                      activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None, name='n8_contract_n4_Conv2D')(n8_contract_n3_Activation_o1)
    n8_contract_n5_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                              epsilon=0.001, center=True,
                                                              scale=True,
                                                              beta_initializer='zeros',
                                                              gamma_initializer='ones',
                                                              moving_mean_initializer='zeros',
                                                              moving_variance_initializer='ones',
                                                              beta_regularizer=None,
                                                              gamma_regularizer=None,
                                                              beta_constraint=None,
                                                              gamma_constraint=None,
                                                              name='n8_contract_n5_BatchNormalization')(n8_contract_n4_Conv2D_o1)
    n8_contract_n6_Activation_o1 = Activation(activation='relu',name='n8_contract_n6_Activation')(n8_contract_n5_BatchNormalization_o1)
    n9_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                      data_format=None, name='n9_MaxPooling2D')(n8_contract_n6_Activation_o1)
    n10_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], strides=[1, 1],
                                       padding='same', data_format=None, dilation_rate=[1, 1],
                                       activation=None, use_bias=True,
                                       kernel_initializer='he_normal', bias_initializer='Zeros',
                                       kernel_regularizer=None, bias_regularizer=None,
                                       activity_regularizer=None, kernel_constraint=None,
                                       bias_constraint=None, name='n10_contract_n1_Conv2D')(n9_MaxPooling2D_o1)
    n10_contract_n2_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                               epsilon=0.001, center=True,
                                                               scale=True,
                                                               beta_initializer='zeros',
                                                               gamma_initializer='ones',
                                                               moving_mean_initializer='zeros',
                                                               moving_variance_initializer='ones',
                                                               beta_regularizer=None,
                                                               gamma_regularizer=None,
                                                               beta_constraint=None,
                                                               gamma_constraint=None,
                                                               name='n10_contract_n2_BatchNormalization')(n10_contract_n1_Conv2D_o1)
    n10_contract_n3_Activation_o1 = Activation(activation='relu', name='n10_contract_n3_Activation')(n10_contract_n2_BatchNormalization_o1)
    n10_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], strides=[1, 1],
                                       padding='same', data_format=None, dilation_rate=[1, 1],
                                       activation=None, use_bias=True,
                                       kernel_initializer='he_normal', bias_initializer='Zeros',
                                       kernel_regularizer=None, bias_regularizer=None,
                                       activity_regularizer=None, kernel_constraint=None,
                                       bias_constraint=None, name='n10_contract_n4_Conv2D')(n10_contract_n3_Activation_o1)
    n10_contract_n5_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                               epsilon=0.001, center=True,
                                                               scale=True,
                                                               beta_initializer='zeros',
                                                               gamma_initializer='ones',
                                                               moving_mean_initializer='zeros',
                                                               moving_variance_initializer='ones',
                                                               beta_regularizer=None,
                                                               gamma_regularizer=None,
                                                               beta_constraint=None,
                                                               gamma_constraint=None,
                                                               name='n10_contract_n5_BatchNormalization')(n10_contract_n4_Conv2D_o1)
    n10_contract_n6_Activation_o1 = Activation(activation='relu',name='n10_contract_n6_Activation')(n10_contract_n5_BatchNormalization_o1)
    n11_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                       data_format=None, name='n11_MaxPooling2D')(n10_contract_n6_Activation_o1)
    n3_bottle_neck_n1_Conv2D_o1 = Conv2D(filters=channel_size*16, kernel_size=[3, 3], strides=[1, 1],
                                         padding='same', data_format=None, dilation_rate=[1, 1],
                                         activation=None, use_bias=True,
                                         kernel_initializer='he_normal',
                                         bias_initializer='Zeros', kernel_regularizer=None,
                                         bias_regularizer=None, activity_regularizer=None,
                                         kernel_constraint=None, bias_constraint=None,
                                         name='n3_bottle_neck_n1_Conv2D')(n11_MaxPooling2D_o1)
    n3_bottle_neck_n2_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                                 epsilon=0.001, center=True,
                                                                 scale=True,
                                                                 beta_initializer='zeros',
                                                                 gamma_initializer='ones',
                                                                 moving_mean_initializer='zeros',
                                                                 moving_variance_initializer='ones',
                                                                 beta_regularizer=None,
                                                                 gamma_regularizer=None,
                                                                 beta_constraint=None,
                                                                 gamma_constraint=None,
                                                                 name='n3_bottle_neck_n2_BatchNormalization')(n3_bottle_neck_n1_Conv2D_o1)
    n3_bottle_neck_n3_Activation_o1 = Activation(activation='relu',name='n3_bottle_neck_n3_Activation')(n3_bottle_neck_n2_BatchNormalization_o1)
    n3_bottle_neck_n4_Conv2D_o1 = Conv2D(filters=channel_size*16, kernel_size=[3, 3], strides=[1, 1],
                                         padding='same', data_format=None, dilation_rate=[1, 1],
                                         activation=None, use_bias=True,
                                         kernel_initializer='he_normal',
                                         bias_initializer='Zeros', kernel_regularizer=None,
                                         bias_regularizer=None, activity_regularizer=None,
                                         kernel_constraint=None, bias_constraint=None,
                                         name='n3_bottle_neck_n4_Conv2D')(n3_bottle_neck_n3_Activation_o1)
    n3_bottle_neck_n5_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                                 epsilon=0.001, center=True,
                                                                 scale=True,
                                                                 beta_initializer='zeros',
                                                                 gamma_initializer='ones',
                                                                 moving_mean_initializer='zeros',
                                                                 moving_variance_initializer='ones',
                                                                 beta_regularizer=None,
                                                                 gamma_regularizer=None,
                                                                 beta_constraint=None,
                                                                 gamma_constraint=None,
                                                                 name='n3_bottle_neck_n5_BatchNormalization')(n3_bottle_neck_n4_Conv2D_o1)
    n3_bottle_neck_n6_Activation_o1 = Activation(activation='relu', name='n3_bottle_neck_n6_Activation')(n3_bottle_neck_n5_BatchNormalization_o1)
    n12_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size*8, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0],
                                             data_format=None, dilation_rate=[1, 1],
                                             activation=None, use_bias=True,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='Zeros', kernel_regularizer=None,
                                             bias_regularizer=None, activity_regularizer=None,
                                             kernel_constraint=None, bias_constraint=None,
                                             name='n12_Conv2DTranspose')(n3_bottle_neck_n6_Activation_o1)
    n4_expand_n1_Concatenate_o1 = concatenate(axis=-1, name='n4_expand_n1_Concatenate')([n12_Conv2DTranspose_o1, n10_contract_n6_Activation_o1])
    n4_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], strides=[1, 1],
                                    padding='same', data_format=None, dilation_rate=[1, 1],
                                    activation=None, use_bias=True,
                                    kernel_initializer='he_normal', bias_initializer='Zeros',
                                    kernel_regularizer=None, bias_regularizer=None,
                                    activity_regularizer=None, kernel_constraint=None,
                                    bias_constraint=None, name='n4_expand_n2_Conv2D')(n4_expand_n1_Concatenate_o1)
    n4_expand_n3_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                            epsilon=0.001, center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            moving_mean_initializer='zeros',
                                                            moving_variance_initializer='ones',
                                                            beta_regularizer=None,
                                                            gamma_regularizer=None,
                                                            beta_constraint=None,
                                                            gamma_constraint=None,
                                                            name='n4_expand_n3_BatchNormalization')(n4_expand_n2_Conv2D_o1)
    n4_expand_n4_Activation_o1 = Activation(activation='relu', name='n4_expand_n4_Activation')(n4_expand_n3_BatchNormalization_o1)
    n4_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[3, 3], strides=[1, 1],
                                    padding='same', data_format=None, dilation_rate=[1, 1],
                                    activation=None, use_bias=True,
                                    kernel_initializer='he_normal', bias_initializer='Zeros',
                                    kernel_regularizer=None, bias_regularizer=None,
                                    activity_regularizer=None, kernel_constraint=None,
                                    bias_constraint=None, name='n4_expand_n5_Conv2D')(n4_expand_n4_Activation_o1)
    n4_expand_n6_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                            epsilon=0.001, center=True,
                                                            scale=True,
                                                            beta_initializer='zeros',
                                                            gamma_initializer='ones',
                                                            moving_mean_initializer='zeros',
                                                            moving_variance_initializer='ones',
                                                            beta_regularizer=None,
                                                            gamma_regularizer=None,
                                                            beta_constraint=None,
                                                            gamma_constraint=None,
                                                            name='n4_expand_n6_BatchNormalization')(n4_expand_n5_Conv2D_o1)
    n4_expand_n7_Activation_o1 = Activation(activation='relu', name='n4_expand_n7_Activation')(n4_expand_n6_BatchNormalization_o1)
    n13_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size*4, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0],
                                             data_format=None, dilation_rate=[1, 1],
                                             activation=None, use_bias=True,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='Zeros', kernel_regularizer=None,
                                             bias_regularizer=None, activity_regularizer=None,
                                             kernel_constraint=None, bias_constraint=None,
                                             name='n13_Conv2DTranspose')(n4_expand_n7_Activation_o1)
    n14_expand_n1_Concatenate_o1 = concatenate(axis=-1, name='n14_expand_n1_Concatenate')([n13_Conv2DTranspose_o1, n8_contract_n6_Activation_o1])
    n14_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], strides=[1, 1],
                                     padding='same', data_format=None, dilation_rate=[1, 1],
                                     activation=None, use_bias=True,
                                     kernel_initializer='he_normal', bias_initializer='Zeros',
                                     kernel_regularizer=None, bias_regularizer=None,
                                     activity_regularizer=None, kernel_constraint=None,
                                     bias_constraint=None, name='n14_expand_n2_Conv2D')(n14_expand_n1_Concatenate_o1)
    n14_expand_n3_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                             epsilon=0.001, center=True,
                                                             scale=True,
                                                             beta_initializer='zeros',
                                                             gamma_initializer='ones',
                                                             moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones',
                                                             beta_regularizer=None,
                                                             gamma_regularizer=None,
                                                             beta_constraint=None,
                                                             gamma_constraint=None,
                                                             name='n14_expand_n3_BatchNormalization')(n14_expand_n2_Conv2D_o1)
    n14_expand_n4_Activation_o1 = Activation(activation='relu',name='n14_expand_n4_Activation')(n14_expand_n3_BatchNormalization_o1)
    n14_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[3, 3], strides=[1, 1],
                                     padding='same', data_format=None, dilation_rate=[1, 1],
                                     activation=None, use_bias=True,
                                     kernel_initializer='he_normal', bias_initializer='Zeros',
                                     kernel_regularizer=None, bias_regularizer=None,
                                     activity_regularizer=None, kernel_constraint=None,
                                     bias_constraint=None, name='n14_expand_n5_Conv2D')(n14_expand_n4_Activation_o1)
    n14_expand_n6_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                             epsilon=0.001, center=True,
                                                             scale=True,
                                                             beta_initializer='zeros',
                                                             gamma_initializer='ones',
                                                             moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones',
                                                             beta_regularizer=None,
                                                             gamma_regularizer=None,
                                                             beta_constraint=None,
                                                             gamma_constraint=None,
                                                             name='n14_expand_n6_BatchNormalization')(n14_expand_n5_Conv2D_o1)
    n14_expand_n7_Activation_o1 = Activation(activation='relu', name='n14_expand_n7_Activation')(n14_expand_n6_BatchNormalization_o1)
    n15_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size*2, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0],
                                             data_format=None, dilation_rate=[1, 1],
                                             activation=None, use_bias=True,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='Zeros', kernel_regularizer=None,
                                             bias_regularizer=None, activity_regularizer=None,
                                             kernel_constraint=None, bias_constraint=None,
                                             name='n15_Conv2DTranspose')(n14_expand_n7_Activation_o1)
    n16_expand_n1_Concatenate_o1 = concatenate(axis=-1, name='n16_expand_n1_Concatenate')([n15_Conv2DTranspose_o1, n6_contract_n6_Activation_o1])
    n16_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], strides=[1, 1],
                                     padding='same', data_format=None, dilation_rate=[1, 1],
                                     activation=None, use_bias=True,
                                     kernel_initializer='he_normal', bias_initializer='Zeros',
                                     kernel_regularizer=None, bias_regularizer=None,
                                     activity_regularizer=None, kernel_constraint=None,
                                     bias_constraint=None, name='n16_expand_n2_Conv2D')(n16_expand_n1_Concatenate_o1)
    n16_expand_n3_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                             epsilon=0.001, center=True,
                                                             scale=True,
                                                             beta_initializer='zeros',
                                                             gamma_initializer='ones',
                                                             moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones',
                                                             beta_regularizer=None,
                                                             gamma_regularizer=None,
                                                             beta_constraint=None,
                                                             gamma_constraint=None,
                                                             name='n16_expand_n3_BatchNormalization')(n16_expand_n2_Conv2D_o1)
    n16_expand_n4_Activation_o1 = Activation(activation='relu', name='n16_expand_n4_Activation')(n16_expand_n3_BatchNormalization_o1)
    n16_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], strides=[1, 1],
                                     padding='same', data_format=None, dilation_rate=[1, 1],
                                     activation=None, use_bias=True,
                                     kernel_initializer='he_normal', bias_initializer='Zeros',
                                     kernel_regularizer=None, bias_regularizer=None,
                                     activity_regularizer=None, kernel_constraint=None,
                                     bias_constraint=None, name='n16_expand_n5_Conv2D')(n16_expand_n4_Activation_o1)
    n16_expand_n6_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                             epsilon=0.001, center=True,
                                                             scale=True,
                                                             beta_initializer='zeros',
                                                             gamma_initializer='ones',
                                                             moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones',
                                                             beta_regularizer=None,
                                                             gamma_regularizer=None,
                                                             beta_constraint=None,
                                                             gamma_constraint=None,
                                                             name='n16_expand_n6_BatchNormalization')(n16_expand_n5_Conv2D_o1)
    n16_expand_n7_Activation_o1 = Activation(activation='relu',
                                             name='n16_expand_n7_Activation')(n16_expand_n6_BatchNormalization_o1)
    n17_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0],
                                             data_format=None, dilation_rate=[1, 1],
                                             activation=None, use_bias=True,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='Zeros', kernel_regularizer=None,
                                             bias_regularizer=None, activity_regularizer=None,
                                             kernel_constraint=None, bias_constraint=None,
                                             name='n17_Conv2DTranspose')(n16_expand_n7_Activation_o1)
    n18_expand_n1_Concatenate_o1 = concatenate(axis=-1, name='n18_expand_n1_Concatenate')([n17_Conv2DTranspose_o1, n2_contract_n6_Activation_o1])
    n18_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], strides=[1, 1],
                                     padding='same', data_format=None, dilation_rate=[1, 1],
                                     activation=None, use_bias=True,
                                     kernel_initializer='he_normal', bias_initializer='Zeros',
                                     kernel_regularizer=None, bias_regularizer=None,
                                     activity_regularizer=None, kernel_constraint=None,
                                     bias_constraint=None, name='n18_expand_n2_Conv2D')(n18_expand_n1_Concatenate_o1)
    n18_expand_n3_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                             epsilon=0.001, center=True,
                                                             scale=True,
                                                             beta_initializer='zeros',
                                                             gamma_initializer='ones',
                                                             moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones',
                                                             beta_regularizer=None,
                                                             gamma_regularizer=None,
                                                             beta_constraint=None,
                                                             gamma_constraint=None,
                                                             name='n18_expand_n3_BatchNormalization')(n18_expand_n2_Conv2D_o1)
    n18_expand_n4_Activation_o1 = Activation(activation='relu', name='n18_expand_n4_Activation')(n18_expand_n3_BatchNormalization_o1)
    n18_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], strides=[1, 1],
                                     padding='same', data_format=None, dilation_rate=[1, 1],
                                     activation=None, use_bias=True,
                                     kernel_initializer='he_normal', bias_initializer='Zeros',
                                     kernel_regularizer=None, bias_regularizer=None,
                                     activity_regularizer=None, kernel_constraint=None,
                                     bias_constraint=None, name='n18_expand_n5_Conv2D')(n18_expand_n4_Activation_o1)
    n18_expand_n6_BatchNormalization_o1 = BatchNormalization(axis=-1, momentum=0.01,
                                                             epsilon=0.001, center=True,
                                                             scale=True,
                                                             beta_initializer='zeros',
                                                             gamma_initializer='ones',
                                                             moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones',
                                                             beta_regularizer=None,
                                                             gamma_regularizer=None,
                                                             beta_constraint=None,
                                                             gamma_constraint=None,
                                                             name='n18_expand_n6_BatchNormalization')(n18_expand_n5_Conv2D_o1)
    n18_expand_n7_Activation_o1 = Activation(activation='relu',name='n18_expand_n7_Activation')(n18_expand_n6_BatchNormalization_o1)
    n19_Conv2D_o1 = Conv2D(filters=2, kernel_size=[1, 1], strides=[1, 1], padding='same',
                           data_format=None, dilation_rate=[1, 1], activation='softmax',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           bias_initializer='Zeros', kernel_regularizer=None,
                           bias_regularizer=None, activity_regularizer=None,
                           kernel_constraint=None, bias_constraint=None, name='n19_Conv2D')(n18_expand_n7_Activation_o1)

    model = Model(inputs=n1_Input_o1, outputs=[n19_Conv2D_o1])


# %%

# Data split for 5 Fold Cross Validation
k = 3
now_fold = 0
kf = KFold(n_splits=k)
KFold(n_splits=k, random_state=None, shuffle=False)

## for debugging
# train_input_split = train_input_split[:5000]
# train_label_split = train_label_split[:5000]

result_dict = {}
for train_index, validation_index in kf.split(train_input_split):
    now_fold += 1
    print('Fold :', now_fold)
    # print("TRAIN:", train_index, "VALIDATION:", validation_index)
    # print("TRAIN:", len(train_index), "VALIDATION:", len(validation_index))
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
    fold_train_label = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_train_label_list]), axis=-1)/255
    print(fold_train_input.shape, fold_train_label.shape)
    fold_validation_input = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_validation_input_list]),axis=-1)
    # fold_validation_label = np.expand_dims(np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_validation_label_list]),axis=-1)/255
    fold_validation_label = np.expand_dims(np.array([[cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY_INV)[1], cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY)[1]] for img in fold_validation_label_list]), axis=-1) / 255

    print(fold_validation_input.shape, fold_validation_label.shape)

    # Train Each Fold
    #model = UNet()
    model = Unet()
    model.summary()
    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[iou_coef, dice_coef])

    history = model.fit(fold_train_input,
                        fold_train_label,
                        validation_data=(fold_validation_input, fold_validation_label),
                        epochs=100,
                        batch_size=16)

# %%



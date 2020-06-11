# Intervertebral Disc Segmentation
## Image Dataset Loader (2D)
## Intervertebral Disc Segmentation

import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Concatenate, DepthwiseConv2D, ZeroPadding2D, Add, AveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from tensorflow.keras.utils import multi_gpu_model


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ## IF GPU RUN, ELIMINATE THIS CODE
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

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


def Unet(input_shape=[256, 256, 1], channel_size=8):
    n1_Input_o1 = Input(shape=input_shape, batch_shape=None, name='n1_Input')

    n2_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n1_Input_o1)
    n2_contract_n2_BatchNormalization_o1 = BatchNormalization()(n2_contract_n1_Conv2D_o1)
    n2_contract_n3_Activation_o1 = Activation(activation='elu')(n2_contract_n2_BatchNormalization_o1)
    n2_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n2_contract_n3_Activation_o1)
    n2_contract_n5_BatchNormalization_o1 = BatchNormalization()(n2_contract_n4_Conv2D_o1)
    n2_contract_n6_Activation_o1 = Activation(activation='elu')(n2_contract_n5_BatchNormalization_o1)

    n5_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n2_contract_n6_Activation_o1)

    n6_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n5_MaxPooling2D_o1)
    n6_contract_n2_BatchNormalization_o1 = BatchNormalization()(n6_contract_n1_Conv2D_o1)
    n6_contract_n3_Activation_o1 = Activation(activation='elu')(n6_contract_n2_BatchNormalization_o1)
    n6_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n6_contract_n3_Activation_o1)
    n6_contract_n5_BatchNormalization_o1 = BatchNormalization()(n6_contract_n4_Conv2D_o1)
    n6_contract_n6_Activation_o1 = Activation(activation='elu')(n6_contract_n5_BatchNormalization_o1)

    n7_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n6_contract_n6_Activation_o1)

    n8_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n7_MaxPooling2D_o1)
    n8_contract_n2_BatchNormalization_o1 = BatchNormalization()(n8_contract_n1_Conv2D_o1)
    n8_contract_n3_Activation_o1 = Activation(activation='elu')(n8_contract_n2_BatchNormalization_o1)
    n8_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n8_contract_n3_Activation_o1)
    n8_contract_n5_BatchNormalization_o1 = BatchNormalization()(n8_contract_n4_Conv2D_o1)
    n8_contract_n6_Activation_o1 = Activation(activation='elu')(n8_contract_n5_BatchNormalization_o1)

    n9_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n8_contract_n6_Activation_o1)

    n10_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                       kernel_initializer='he_normal')(n9_MaxPooling2D_o1)
    n10_contract_n2_BatchNormalization_o1 = BatchNormalization()(n10_contract_n1_Conv2D_o1)
    n10_contract_n3_Activation_o1 = Activation(activation='elu')(n10_contract_n2_BatchNormalization_o1)
    n10_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                       kernel_initializer='he_normal')(n10_contract_n3_Activation_o1)
    n10_contract_n5_BatchNormalization_o1 = BatchNormalization()(n10_contract_n4_Conv2D_o1)
    n10_contract_n6_Activation_o1 = Activation(activation='elu')(n10_contract_n5_BatchNormalization_o1)

    n11_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n10_contract_n6_Activation_o1)

    n3_bottle_neck_n1_Conv2D_o1 = Conv2D(filters=channel_size * 16, kernel_size=[3, 3], padding='same',
                                         kernel_initializer='he_normal')(n11_MaxPooling2D_o1)
    n3_bottle_neck_n2_BatchNormalization_o1 = BatchNormalization()(n3_bottle_neck_n1_Conv2D_o1)
    n3_bottle_neck_n3_Activation_o1 = Activation(activation='elu')(n3_bottle_neck_n2_BatchNormalization_o1)
    n3_bottle_neck_n4_Conv2D_o1 = Conv2D(filters=channel_size * 16, kernel_size=[3, 3], padding='same',
                                         kernel_initializer='he_normal')(n3_bottle_neck_n3_Activation_o1)
    n3_bottle_neck_n5_BatchNormalization_o1 = BatchNormalization()(n3_bottle_neck_n4_Conv2D_o1)
    n3_bottle_neck_n6_Activation_o1 = Activation(activation='elu')(n3_bottle_neck_n5_BatchNormalization_o1)

    n12_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size * 8, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(
        n3_bottle_neck_n6_Activation_o1)

    n4_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n12_Conv2DTranspose_o1, n10_contract_n6_Activation_o1])
    n4_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                    kernel_initializer='he_normal')(n4_expand_n1_Concatenate_o1)
    n4_expand_n3_BatchNormalization_o1 = BatchNormalization()(n4_expand_n2_Conv2D_o1)
    n4_expand_n4_Activation_o1 = Activation(activation='elu')(n4_expand_n3_BatchNormalization_o1)
    n4_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                    kernel_initializer='he_normal')(n4_expand_n4_Activation_o1)
    n4_expand_n6_BatchNormalization_o1 = BatchNormalization()(n4_expand_n5_Conv2D_o1)
    n4_expand_n7_Activation_o1 = Activation(activation='elu')(n4_expand_n6_BatchNormalization_o1)

    n13_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size * 4, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(
        n4_expand_n7_Activation_o1)

    n14_expand_n1_Concatenate_o1 = Concatenate(axis=-1, name='n14_expand_n1_Concatenate')(
        [n13_Conv2DTranspose_o1, n8_contract_n6_Activation_o1])
    n14_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n14_expand_n1_Concatenate_o1)
    n14_expand_n3_BatchNormalization_o1 = BatchNormalization()(n14_expand_n2_Conv2D_o1)
    n14_expand_n4_Activation_o1 = Activation(activation='elu')(n14_expand_n3_BatchNormalization_o1)
    n14_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n14_expand_n4_Activation_o1)
    n14_expand_n6_BatchNormalization_o1 = BatchNormalization()(n14_expand_n5_Conv2D_o1)
    n14_expand_n7_Activation_o1 = Activation(activation='elu')(n14_expand_n6_BatchNormalization_o1)

    n15_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size * 2, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(
        n14_expand_n7_Activation_o1)

    n16_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n15_Conv2DTranspose_o1, n6_contract_n6_Activation_o1])
    n16_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n16_expand_n1_Concatenate_o1)
    n16_expand_n3_BatchNormalization_o1 = BatchNormalization()(n16_expand_n2_Conv2D_o1)
    n16_expand_n4_Activation_o1 = Activation(activation='elu')(n16_expand_n3_BatchNormalization_o1)
    n16_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n16_expand_n4_Activation_o1)
    n16_expand_n6_BatchNormalization_o1 = BatchNormalization()(n16_expand_n5_Conv2D_o1)
    n16_expand_n7_Activation_o1 = Activation(activation='elu')(n16_expand_n6_BatchNormalization_o1)

    n17_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size, kernel_size=[2, 2], strides=[2, 2], padding='valid',
                                             output_padding=[0, 0], dilation_rate=[1, 1])(n16_expand_n7_Activation_o1)

    n18_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n17_Conv2DTranspose_o1, n2_contract_n6_Activation_o1])
    n18_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n18_expand_n1_Concatenate_o1)
    n18_expand_n3_BatchNormalization_o1 = BatchNormalization()(n18_expand_n2_Conv2D_o1)
    n18_expand_n4_Activation_o1 = Activation(activation='elu')(n18_expand_n3_BatchNormalization_o1)
    n18_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n18_expand_n4_Activation_o1)
    n18_expand_n6_BatchNormalization_o1 = BatchNormalization()(n18_expand_n5_Conv2D_o1)
    n18_expand_n7_Activation_o1 = Activation(activation='elu')(n18_expand_n6_BatchNormalization_o1)

    n19_Conv2D_o1 = Conv2D(filters=2, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='softmax',
                           kernel_initializer='he_normal')(n18_expand_n7_Activation_o1)

    model = Model(inputs=n1_Input_o1, outputs=[n19_Conv2D_o1])
    return model


#######################
def DeeplabV3(input_shape=[256, 256, 1], channel_size=8):
    n1_Input_o1 = Input(input_shape)
    n2_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], strides=[2, 2], padding='same')(n1_Input_o1)
    n3_BatchNormalization_o1 = BatchNormalization()(n2_Conv2D_o1)
    n4_Activation_o1 = Activation(activation='relu')(n3_BatchNormalization_o1)
    n5_Conv2D_o1 = Conv2D(filters=channel_size*2, kernel_size=[3, 3], strides=[1, 1], padding='same')(n4_Activation_o1)
    n6_BatchNormalization_o1 = BatchNormalization()(n5_Conv2D_o1)
    n7_Activation_o1 = Activation(activation='relu')(n6_BatchNormalization_o1)
    n8_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n7_Activation_o1)
    n8_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[1, 1], strides=[2, 2], padding='same')(n8_Xception_Block_n1_Activation_o1)
    n8_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n2_Conv2D_o1)
    n8_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n8_Xception_Block_n3_BatchNormalization_o1)
    n8_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n8_Xception_Block_n1_Activation_o1)
    n8_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n8_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n8_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n8_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n8_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n8_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n8_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n8_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n8_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n8_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n8_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n8_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n8_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n8_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n8_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n8_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n8_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n8_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n8_Xception_Block_n8_SepConv_BN_n8_ZeroPadding2D_o1 = ZeroPadding2D(padding=[[1, 1], [1, 1]])(n8_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n8_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n8_Xception_Block_n8_SepConv_BN_n8_ZeroPadding2D_o1)
    n8_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding='valid', depthwise_initializer='he_normal')(n8_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n8_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n8_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n8_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*4, kernel_size=[1, 1], strides=[1, 1], padding='same')(n8_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n8_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n8_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n8_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n8_Xception_Block_n5_Add_o1 = Add()([n8_Xception_Block_n4_Activation_o1, n8_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n9_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n8_Xception_Block_n5_Add_o1)
    n9_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[2, 2], padding='same')(n9_Xception_Block_n1_Activation_o1)
    n9_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n2_Conv2D_o1)
    n9_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n9_Xception_Block_n3_BatchNormalization_o1)
    n9_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n9_Xception_Block_n1_Activation_o1)
    n9_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n9_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n9_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n9_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n9_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n9_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n9_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n9_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n9_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n9_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n9_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n9_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n9_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n9_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n9_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n9_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n9_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n9_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n9_Xception_Block_n8_SepConv_BN_n8_ZeroPadding2D_o1 = ZeroPadding2D(padding=[[1, 1], [1, 1]])(n9_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n9_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n9_Xception_Block_n8_SepConv_BN_n8_ZeroPadding2D_o1)
    n9_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding='valid', depthwise_initializer='he_normal')(n9_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n9_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n9_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n9_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n9_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n9_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n9_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n9_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n9_Xception_Block_n5_Add_o1 = Add()([n9_Xception_Block_n4_Activation_o1, n9_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n10_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n9_Xception_Block_n5_Add_o1)
    n10_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[2, 2], padding='same')(n10_Xception_Block_n1_Activation_o1)
    n10_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n2_Conv2D_o1)
    n10_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n10_Xception_Block_n3_BatchNormalization_o1)
    n10_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n10_Xception_Block_n1_Activation_o1)
    n10_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n10_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n10_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n10_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n10_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n10_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n10_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n10_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n10_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n10_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n10_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n10_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n10_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n10_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n10_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n10_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n10_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n10_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n10_Xception_Block_n8_SepConv_BN_n8_ZeroPadding2D_o1 = ZeroPadding2D(padding=[[1, 1], [1, 1]])(n10_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n39_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n9_Xception_Block_n5_Add_o1)
    n40_BatchNormalization_o1 = BatchNormalization()(n39_Conv2D_o1)
    n41_Activation_o1 = Activation(activation='relu')(n40_BatchNormalization_o1)
    n10_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n10_Xception_Block_n8_SepConv_BN_n8_ZeroPadding2D_o1)
    n10_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding='valid', depthwise_initializer='he_normal')(n10_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n10_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n10_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n10_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n10_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n10_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n10_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n10_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n10_Xception_Block_n5_Add_o1 = Add()([n10_Xception_Block_n4_Activation_o1, n10_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n11_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n10_Xception_Block_n5_Add_o1)
    n11_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n11_Xception_Block_n1_Activation_o1)
    n11_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n2_Conv2D_o1)
    n11_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n11_Xception_Block_n3_BatchNormalization_o1)
    n11_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n11_Xception_Block_n1_Activation_o1)
    n11_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n11_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n11_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n11_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n11_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n11_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n11_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n11_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n11_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n11_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n11_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n11_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n11_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n11_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n11_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n11_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n11_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n11_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n11_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n11_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n11_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n11_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n11_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n11_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n11_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n11_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n11_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n11_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n11_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n11_Xception_Block_n5_Add_o1 = Add()([n11_Xception_Block_n4_Activation_o1, n11_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n12_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n11_Xception_Block_n5_Add_o1)
    n12_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n12_Xception_Block_n1_Activation_o1)
    n12_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n2_Conv2D_o1)
    n12_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n12_Xception_Block_n3_BatchNormalization_o1)
    n12_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n12_Xception_Block_n1_Activation_o1)
    n12_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n12_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n12_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n12_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n12_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n12_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n12_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n12_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n12_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n12_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n12_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n12_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n12_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n12_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n12_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n12_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n12_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n12_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n12_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n12_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n12_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n12_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n12_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n12_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n12_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n12_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n12_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n12_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n12_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n12_Xception_Block_n5_Add_o1 = Add()([n12_Xception_Block_n4_Activation_o1, n12_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n13_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n12_Xception_Block_n5_Add_o1)
    n13_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n13_Xception_Block_n1_Activation_o1)
    n13_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n2_Conv2D_o1)
    n13_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n13_Xception_Block_n3_BatchNormalization_o1)
    n13_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n13_Xception_Block_n1_Activation_o1)
    n13_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n13_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n13_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n13_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n13_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n13_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n13_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n13_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n13_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n13_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n13_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n13_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n13_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n13_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n13_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n13_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n13_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n13_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n13_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n13_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n13_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n13_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n13_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n13_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n13_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n13_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n13_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n13_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n13_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n13_Xception_Block_n5_Add_o1 = Add()([n13_Xception_Block_n4_Activation_o1, n13_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n14_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n13_Xception_Block_n5_Add_o1)
    n14_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n14_Xception_Block_n1_Activation_o1)
    n14_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n2_Conv2D_o1)
    n14_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n14_Xception_Block_n3_BatchNormalization_o1)
    n14_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n14_Xception_Block_n1_Activation_o1)
    n14_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n14_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n14_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n14_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n14_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n14_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n14_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n14_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n14_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n14_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n14_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n14_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n14_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n14_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n14_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n14_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n14_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n14_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n14_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n14_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n14_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n14_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n14_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n14_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n14_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n14_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n14_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n14_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n14_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n14_Xception_Block_n5_Add_o1 = Add()([n14_Xception_Block_n4_Activation_o1, n14_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n15_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n14_Xception_Block_n5_Add_o1)
    n15_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n15_Xception_Block_n1_Activation_o1)
    n15_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n2_Conv2D_o1)
    n15_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n15_Xception_Block_n3_BatchNormalization_o1)
    n15_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n15_Xception_Block_n1_Activation_o1)
    n15_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n15_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n15_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n15_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n15_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n15_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n15_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n15_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n15_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n15_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n15_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n15_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n15_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n15_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n15_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n15_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n15_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n15_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n15_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n15_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n15_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n15_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n15_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n15_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n15_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n15_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n15_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n15_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n15_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n15_Xception_Block_n5_Add_o1 = Add()([n15_Xception_Block_n4_Activation_o1, n15_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n16_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n15_Xception_Block_n5_Add_o1)
    n16_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n16_Xception_Block_n1_Activation_o1)
    n16_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n2_Conv2D_o1)
    n16_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n16_Xception_Block_n3_BatchNormalization_o1)
    n16_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n16_Xception_Block_n1_Activation_o1)
    n16_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n16_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n16_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n16_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n16_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n16_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n16_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n16_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n16_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n16_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n16_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n16_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n16_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n16_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n16_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n16_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n16_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n16_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n16_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n16_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n16_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n16_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n16_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n16_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n16_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n16_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n16_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n16_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n16_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n16_Xception_Block_n5_Add_o1 = Add()([n16_Xception_Block_n4_Activation_o1, n16_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n17_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n16_Xception_Block_n5_Add_o1)
    n17_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n17_Xception_Block_n1_Activation_o1)
    n17_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n2_Conv2D_o1)
    n17_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n17_Xception_Block_n3_BatchNormalization_o1)
    n17_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n17_Xception_Block_n1_Activation_o1)
    n17_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n17_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n17_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n17_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n17_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n17_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n17_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n17_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n17_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n17_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n17_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n17_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n17_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n17_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n17_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n17_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n17_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n17_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n17_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n17_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n17_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n17_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n17_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n17_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n17_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n17_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n17_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n17_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n17_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n17_Xception_Block_n5_Add_o1 = Add()([n17_Xception_Block_n4_Activation_o1, n17_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n18_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n17_Xception_Block_n5_Add_o1)
    n18_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n18_Xception_Block_n1_Activation_o1)
    n18_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n2_Conv2D_o1)
    n18_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n18_Xception_Block_n3_BatchNormalization_o1)
    n18_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n18_Xception_Block_n1_Activation_o1)
    n18_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n18_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n18_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n18_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n18_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n18_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n18_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n18_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n18_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n18_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n18_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n18_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n18_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n18_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n18_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n18_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n18_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n18_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n18_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n18_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n18_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n18_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n18_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n18_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n18_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n18_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n18_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n18_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n18_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n18_Xception_Block_n5_Add_o1 = Add()([n18_Xception_Block_n4_Activation_o1, n18_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n19_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n18_Xception_Block_n5_Add_o1)
    n19_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n19_Xception_Block_n1_Activation_o1)
    n19_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n2_Conv2D_o1)
    n19_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n19_Xception_Block_n3_BatchNormalization_o1)
    n19_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n19_Xception_Block_n1_Activation_o1)
    n19_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n19_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n19_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n19_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n19_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n19_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n19_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n19_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n19_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n19_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n19_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n19_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n19_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n19_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n19_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n19_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n19_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n19_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n19_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n19_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n19_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n19_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n19_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n19_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n19_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n19_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n19_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n19_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n19_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n19_Xception_Block_n5_Add_o1 = Add()([n19_Xception_Block_n4_Activation_o1, n19_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n20_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n19_Xception_Block_n5_Add_o1)
    n20_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n20_Xception_Block_n1_Activation_o1)
    n20_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n2_Conv2D_o1)
    n20_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n20_Xception_Block_n3_BatchNormalization_o1)
    n20_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n20_Xception_Block_n1_Activation_o1)
    n20_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n20_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n20_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n20_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n20_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n20_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n20_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n20_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n20_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n20_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n20_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n20_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n20_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n20_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n20_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n20_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n20_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n20_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n20_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n20_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n20_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n20_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n20_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n20_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n20_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n20_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n20_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n20_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n20_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n20_Xception_Block_n5_Add_o1 = Add()([n20_Xception_Block_n4_Activation_o1, n20_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n21_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n20_Xception_Block_n5_Add_o1)
    n21_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n21_Xception_Block_n1_Activation_o1)
    n21_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n2_Conv2D_o1)
    n21_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n21_Xception_Block_n3_BatchNormalization_o1)
    n21_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n21_Xception_Block_n1_Activation_o1)
    n21_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n21_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n21_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n21_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n21_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n21_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n21_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n21_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n21_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n21_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n21_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n21_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n21_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n21_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n21_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n21_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n21_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n21_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n21_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n21_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n21_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n21_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n21_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n21_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n21_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n21_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n21_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n21_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n21_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n21_Xception_Block_n5_Add_o1 = Add()([n21_Xception_Block_n4_Activation_o1, n21_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n22_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n21_Xception_Block_n5_Add_o1)
    n22_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n22_Xception_Block_n1_Activation_o1)
    n22_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n2_Conv2D_o1)
    n22_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n22_Xception_Block_n3_BatchNormalization_o1)
    n22_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n22_Xception_Block_n1_Activation_o1)
    n22_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n22_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n22_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n22_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n22_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n22_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n22_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n22_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n22_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n22_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n22_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n22_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n22_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n22_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n22_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n22_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n22_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n22_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n22_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n22_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n22_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n22_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n22_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n22_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n22_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n22_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n22_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n22_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n22_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n22_Xception_Block_n5_Add_o1 = Add()([n22_Xception_Block_n4_Activation_o1, n22_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n23_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n22_Xception_Block_n5_Add_o1)
    n23_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n23_Xception_Block_n1_Activation_o1)
    n23_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n2_Conv2D_o1)
    n23_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n23_Xception_Block_n3_BatchNormalization_o1)
    n23_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n23_Xception_Block_n1_Activation_o1)
    n23_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n23_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n23_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n23_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n23_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n23_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n23_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n23_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n23_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n23_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n23_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n23_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n23_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n23_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n23_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n23_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n23_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n23_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n23_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n23_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n23_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n23_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n23_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n23_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n23_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n23_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n23_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n23_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n23_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n23_Xception_Block_n5_Add_o1 = Add()([n23_Xception_Block_n4_Activation_o1, n23_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n24_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n23_Xception_Block_n5_Add_o1)
    n24_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n24_Xception_Block_n1_Activation_o1)
    n24_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n2_Conv2D_o1)
    n24_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n24_Xception_Block_n3_BatchNormalization_o1)
    n24_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n24_Xception_Block_n1_Activation_o1)
    n24_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n24_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n24_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n24_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n24_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n24_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n24_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n24_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n24_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n24_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n24_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n24_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n24_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n24_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n24_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n24_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n24_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n24_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n24_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n24_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n24_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n24_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n24_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n24_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n24_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n24_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n24_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n24_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n24_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n24_Xception_Block_n5_Add_o1 = Add()([n24_Xception_Block_n4_Activation_o1, n24_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n25_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n24_Xception_Block_n5_Add_o1)
    n25_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n25_Xception_Block_n1_Activation_o1)
    n25_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n2_Conv2D_o1)
    n25_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n25_Xception_Block_n3_BatchNormalization_o1)
    n25_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n25_Xception_Block_n1_Activation_o1)
    n25_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n25_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n25_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n25_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n25_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n25_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n25_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n25_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n25_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n25_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n25_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n25_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n25_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n25_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n25_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n25_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n25_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n25_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n25_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n25_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n25_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n25_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n25_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n25_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n25_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n25_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n25_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n25_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n25_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n25_Xception_Block_n5_Add_o1 = Add()([n25_Xception_Block_n4_Activation_o1, n25_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n26_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n25_Xception_Block_n5_Add_o1)
    n26_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n26_Xception_Block_n1_Activation_o1)
    n26_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n2_Conv2D_o1)
    n26_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n26_Xception_Block_n3_BatchNormalization_o1)
    n26_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n26_Xception_Block_n1_Activation_o1)
    n26_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n26_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n26_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n6_SepConv_BN_n2_DepthwiseConv2D_o1)
    n26_Xception_Block_n6_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n6_SepConv_BN_n3_BatchNormalization_o1)
    n26_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n26_Xception_Block_n6_SepConv_BN_n4_Activation_o1)
    n26_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n6_SepConv_BN_n5_Conv2D_o1)
    n26_Xception_Block_n6_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n6_SepConv_BN_n6_BatchNormalization_o1)
    n26_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n26_Xception_Block_n6_SepConv_BN_n7_Activation_o1)
    n26_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n26_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n26_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n7_SepConv_BN_n2_DepthwiseConv2D_o1)
    n26_Xception_Block_n7_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n7_SepConv_BN_n3_BatchNormalization_o1)
    n26_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n26_Xception_Block_n7_SepConv_BN_n4_Activation_o1)
    n26_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n7_SepConv_BN_n5_Conv2D_o1)
    n26_Xception_Block_n7_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n7_SepConv_BN_n6_BatchNormalization_o1)
    n26_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n26_Xception_Block_n7_SepConv_BN_n7_Activation_o1)
    n26_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depthwise_initializer='he_normal')(n26_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n26_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n8_SepConv_BN_n2_DepthwiseConv2D_o1)
    n26_Xception_Block_n8_SepConv_BN_n4_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n8_SepConv_BN_n3_BatchNormalization_o1)
    n26_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='same')(n26_Xception_Block_n8_SepConv_BN_n4_Activation_o1)
    n26_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1 = BatchNormalization()(n26_Xception_Block_n8_SepConv_BN_n5_Conv2D_o1)
    n26_Xception_Block_n8_SepConv_BN_n7_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n8_SepConv_BN_n6_BatchNormalization_o1)
    n26_Xception_Block_n5_Add_o1 = Add()([n26_Xception_Block_n4_Activation_o1, n26_Xception_Block_n8_SepConv_BN_n7_Activation_o1])
    n27_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n26_Xception_Block_n5_Add_o1)
    n27_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*32, kernel_size=[1, 1], strides=[1, 1], padding='same')(n27_Xception_Block_n1_Activation_o1)
    n27_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n2_Conv2D_o1)
    n27_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n27_Xception_Block_n3_BatchNormalization_o1)
    n27_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n27_Xception_Block_n1_Activation_o1)
    n27_Xception_Block_n6_SepConv_BN_n7_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, data_format=None, dilation_rate=[2, 2], activation=None, use_bias=True, depthwise_initializer='he_normal')(n27_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n27_Xception_Block_n6_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n6_SepConv_BN_n7_DepthwiseConv2D_o1)
    n27_Xception_Block_n6_SepConv_BN_n3_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n6_SepConv_BN_n2_BatchNormalization_o1)
    n27_Xception_Block_n6_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*12, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n27_Xception_Block_n6_SepConv_BN_n3_Activation_o1)
    n27_Xception_Block_n6_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n6_SepConv_BN_n4_Conv2D_o1)
    n27_Xception_Block_n6_SepConv_BN_n6_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n6_SepConv_BN_n5_BatchNormalization_o1)
    n27_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n27_Xception_Block_n6_SepConv_BN_n6_Activation_o1)
    n27_Xception_Block_n7_SepConv_BN_n7_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, data_format=None, dilation_rate=[2, 2], activation=None, use_bias=True, depthwise_initializer='he_normal')(n27_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n27_Xception_Block_n7_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n7_SepConv_BN_n7_DepthwiseConv2D_o1)
    n27_Xception_Block_n7_SepConv_BN_n3_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n7_SepConv_BN_n2_BatchNormalization_o1)
    n27_Xception_Block_n7_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*32, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n27_Xception_Block_n7_SepConv_BN_n3_Activation_o1)
    n27_Xception_Block_n7_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n7_SepConv_BN_n4_Conv2D_o1)
    n27_Xception_Block_n7_SepConv_BN_n6_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n7_SepConv_BN_n5_BatchNormalization_o1)
    n27_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='relu')(n27_Xception_Block_n7_SepConv_BN_n6_Activation_o1)
    n27_Xception_Block_n8_SepConv_BN_n7_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, data_format=None, dilation_rate=[2, 2], activation=None, use_bias=True, depthwise_initializer='he_normal')(n27_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n27_Xception_Block_n8_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n8_SepConv_BN_n7_DepthwiseConv2D_o1)
    n27_Xception_Block_n8_SepConv_BN_n3_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n8_SepConv_BN_n2_BatchNormalization_o1)
    n27_Xception_Block_n8_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*32, kernel_size=[1, 1], strides=[1, 1], padding='same')(n27_Xception_Block_n8_SepConv_BN_n3_Activation_o1)
    n27_Xception_Block_n8_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n27_Xception_Block_n8_SepConv_BN_n4_Conv2D_o1)
    n27_Xception_Block_n8_SepConv_BN_n6_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n8_SepConv_BN_n5_BatchNormalization_o1)
    n27_Xception_Block_n5_Add_o1 = Add()([n27_Xception_Block_n4_Activation_o1, n27_Xception_Block_n8_SepConv_BN_n6_Activation_o1])
    n28_Xception_Block_n1_Activation_o1 = Activation(activation='linear')(n27_Xception_Block_n5_Add_o1)
    n28_Xception_Block_n2_Conv2D_o1 = Conv2D(filters=channel_size*64, kernel_size=[1, 1], strides=[1, 1], padding='same')(n28_Xception_Block_n1_Activation_o1)
    n28_Xception_Block_n3_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n2_Conv2D_o1)
    n28_Xception_Block_n4_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n3_BatchNormalization_o1)
    n28_Xception_Block_n6_SepConv_BN_n1_Activation_o1 = Activation(activation='linear')(n28_Xception_Block_n1_Activation_o1)
    n28_Xception_Block_n6_SepConv_BN_n7_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, data_format=None, dilation_rate=[4, 4], activation=None, use_bias=True, depthwise_initializer='he_normal')(n28_Xception_Block_n6_SepConv_BN_n1_Activation_o1)
    n28_Xception_Block_n6_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n6_SepConv_BN_n7_DepthwiseConv2D_o1)
    n28_Xception_Block_n6_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n6_SepConv_BN_n2_BatchNormalization_o1)
    n28_Xception_Block_n6_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*48, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n28_Xception_Block_n6_SepConv_BN_n3_Activation_o1)
    n28_Xception_Block_n6_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n6_SepConv_BN_n4_Conv2D_o1)
    n28_Xception_Block_n6_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n6_SepConv_BN_n5_BatchNormalization_o1)
    n28_Xception_Block_n7_SepConv_BN_n1_Activation_o1 = Activation(activation='linear')(n28_Xception_Block_n6_SepConv_BN_n6_Activation_o1)
    n28_Xception_Block_n7_SepConv_BN_n7_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, data_format=None, dilation_rate=[4, 4], activation=None, use_bias=True, depthwise_initializer='he_normal')(n28_Xception_Block_n7_SepConv_BN_n1_Activation_o1)
    n28_Xception_Block_n7_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n7_SepConv_BN_n7_DepthwiseConv2D_o1)
    n28_Xception_Block_n7_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n7_SepConv_BN_n2_BatchNormalization_o1)
    n28_Xception_Block_n7_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*48, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n28_Xception_Block_n7_SepConv_BN_n3_Activation_o1)
    n28_Xception_Block_n7_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n7_SepConv_BN_n4_Conv2D_o1)
    n28_Xception_Block_n7_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n7_SepConv_BN_n5_BatchNormalization_o1)
    n28_Xception_Block_n8_SepConv_BN_n1_Activation_o1 = Activation(activation='linear')(n28_Xception_Block_n7_SepConv_BN_n6_Activation_o1)
    n28_Xception_Block_n8_SepConv_BN_n7_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, data_format=None, dilation_rate=[4, 4], activation=None, use_bias=True, depthwise_initializer='he_normal')(n28_Xception_Block_n8_SepConv_BN_n1_Activation_o1)
    n28_Xception_Block_n8_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n8_SepConv_BN_n7_DepthwiseConv2D_o1)
    n28_Xception_Block_n8_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n8_SepConv_BN_n2_BatchNormalization_o1)
    n28_Xception_Block_n8_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*64, kernel_size=[1, 1], strides=[1, 1], padding='same')(n28_Xception_Block_n8_SepConv_BN_n3_Activation_o1)
    n28_Xception_Block_n8_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n28_Xception_Block_n8_SepConv_BN_n4_Conv2D_o1)
    n28_Xception_Block_n8_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n28_Xception_Block_n8_SepConv_BN_n5_BatchNormalization_o1)
    n28_Xception_Block_n5_Add_o1 = Add()([n28_Xception_Block_n4_Activation_o1, n28_Xception_Block_n8_SepConv_BN_n6_Activation_o1])
    n29_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n28_Xception_Block_n5_Add_o1)
    n30_SepConv_BN_n1_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, dilation_rate=[6, 6],depthwise_initializer='he_normal')(n28_Xception_Block_n5_Add_o1)
    n30_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n30_SepConv_BN_n1_DepthwiseConv2D_o1)
    n30_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n30_SepConv_BN_n2_BatchNormalization_o1)
    n30_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n30_SepConv_BN_n3_Activation_o1)
    n30_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n30_SepConv_BN_n4_Conv2D_o1)
    n30_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n30_SepConv_BN_n5_BatchNormalization_o1)
    n31_SepConv_BN_n1_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, dilation_rate=[12, 12], depthwise_initializer='he_normal')(n28_Xception_Block_n5_Add_o1)
    n31_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n31_SepConv_BN_n1_DepthwiseConv2D_o1)
    n31_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n31_SepConv_BN_n2_BatchNormalization_o1)
    n31_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n31_SepConv_BN_n3_Activation_o1)
    n31_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n31_SepConv_BN_n4_Conv2D_o1)
    n31_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n31_SepConv_BN_n5_BatchNormalization_o1)
    n32_SepConv_BN_n1_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, dilation_rate=[18, 18], depthwise_initializer='he_normal')(n28_Xception_Block_n5_Add_o1)
    n32_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n32_SepConv_BN_n1_DepthwiseConv2D_o1)
    n32_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n32_SepConv_BN_n2_BatchNormalization_o1)
    n32_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n32_SepConv_BN_n3_Activation_o1)
    n32_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n32_SepConv_BN_n4_Conv2D_o1)
    n32_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n32_SepConv_BN_n5_BatchNormalization_o1)
    n33_Image_Pooling_n1_AveragePooling2D_o1 = AveragePooling2D(pool_size=[16, 16], strides=[2, 2], padding='valid')(n28_Xception_Block_n5_Add_o1)
    n33_Image_Pooling_n2_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n33_Image_Pooling_n1_AveragePooling2D_o1)
    n33_Image_Pooling_n3_BatchNormalization_o1 = BatchNormalization()(n33_Image_Pooling_n2_Conv2D_o1)
    n33_Image_Pooling_n4_Activation_o1 = Activation(activation='relu')(n33_Image_Pooling_n3_BatchNormalization_o1)
    n33_Image_Pooling_n5_Resize_o1 = tf.image.resize(n33_Image_Pooling_n4_Activation_o1, size=[16, 16], method='bilinear', antialias=True, preserve_aspect_ratio=True)
    n34_Concatenate_o1 = Concatenate(axis=-1)([n29_Conv2D_o1, n30_SepConv_BN_n6_Activation_o1, n31_SepConv_BN_n6_Activation_o1, n32_SepConv_BN_n6_Activation_o1, n33_Image_Pooling_n5_Resize_o1])
    n35_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='valid')(n34_Concatenate_o1)
    n36_BatchNormalization_o1 = BatchNormalization()(n35_Conv2D_o1)
    n37_Activation_o1 = Activation(activation='relu')(n36_BatchNormalization_o1)
    n38_Dropout_o1 = Dropout(rate=0.1)(n37_Activation_o1)
    n42_Resize_o1 = tf.image.resize(n38_Dropout_o1, size=[32, 32], method='bilinear', antialias=True, preserve_aspect_ratio=True)
    n43_Concatenate_o1 = Concatenate(axis=-1)([n41_Activation_o1, n42_Resize_o1])
    n44_SepConv_BN_n1_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, dilation_rate=[6, 6], depthwise_initializer='he_normal')(n43_Concatenate_o1)
    n44_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n44_SepConv_BN_n1_DepthwiseConv2D_o1)
    n44_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n44_SepConv_BN_n2_BatchNormalization_o1)
    n44_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n44_SepConv_BN_n3_Activation_o1)
    n44_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n44_SepConv_BN_n4_Conv2D_o1)
    n44_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n44_SepConv_BN_n5_BatchNormalization_o1)
    n45_SepConv_BN_n1_DepthwiseConv2D_o1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding='same', depth_multiplier=1, dilation_rate=[6, 6], depthwise_initializer='he_normal')(n44_SepConv_BN_n6_Activation_o1)
    n45_SepConv_BN_n2_BatchNormalization_o1 = BatchNormalization()(n45_SepConv_BN_n1_DepthwiseConv2D_o1)
    n45_SepConv_BN_n3_Activation_o1 = Activation(activation='relu')(n45_SepConv_BN_n2_BatchNormalization_o1)
    n45_SepConv_BN_n4_Conv2D_o1 = Conv2D(filters=channel_size*8, kernel_size=[1, 1], strides=[1, 1], padding='same')(n45_SepConv_BN_n3_Activation_o1)
    n45_SepConv_BN_n5_BatchNormalization_o1 = BatchNormalization()(n45_SepConv_BN_n4_Conv2D_o1)
    n45_SepConv_BN_n6_Activation_o1 = Activation(activation='relu')(n45_SepConv_BN_n5_BatchNormalization_o1)
    n46_Resize_o1 = tf.image.resize(n45_SepConv_BN_n6_Activation_o1, size=[input_shape[0], input_shape[1]], method='bilinear', antialias=True, preserve_aspect_ratio=True)
    n47_Conv2D_o1 = Conv2D(filters=2, kernel_size=[1, 1], strides=[1, 1], padding='valid', activation='softmax')(n46_Resize_o1)

    model = Model(inputs=[n1_Input_o1], outputs=[n47_Conv2D_o1])

    return model

#
# # Data split for 5 Fold Cross Validation
# k = 3
# now_fold = 0
# kf = KFold(n_splits=k)
# KFold(n_splits=k, random_state=None, shuffle=False)
#
# ## for debugging
# # train_input_split = train_input_split[:1000]
# # train_label_split = train_label_split[:1000]
#
# result_dict = {}
# for train_index, validation_index in kf.split(train_input_split):
#     now_fold += 1
#     print('Fold :', now_fold)
#     # data random shuffle for regularization
#     np.random.shuffle(train_index)
#
#     # data split
#     fold_train_input_list = train_input_split[train_index]
#     fold_train_label_list = train_label_split[train_index]
#     fold_validation_input_list = train_input_split[validation_index]
#     fold_validation_label_list = train_input_split[validation_index]
#
#     # image data read by using opencv
#     fold_train_input_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_train_input_list]
#     fold_train_label_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_train_label_list]
#     fold_validation_input_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_validation_input_list]
#     fold_validation_label_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in fold_validation_label_list]
#
#     # image data resizing by using opencv
#     fold_train_input = np.expand_dims(
#         np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_train_input_list]), axis=-1)
#     fold_train_label = np.transpose(np.array([[cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA),
#                                                              50, 255, cv2.THRESH_BINARY_INV)[1],
#                                                cv2.threshold(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA),
#                                                              50, 255, cv2.THRESH_BINARY)[1]] for img in
#                                               fold_train_label_list]), axes=[0, 2, 3, 1]) / 255
#     print(fold_train_input.shape, fold_train_label.shape)
#     fold_validation_input = np.expand_dims(
#         np.array([cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in fold_validation_input_list]),
#         axis=-1)
#     fold_validation_label = np.transpose(np.array([[cv2.threshold(
#         cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY_INV)[1], cv2.threshold(
#         cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), 50, 255, cv2.THRESH_BINARY)[1]] for img in
#                                                    fold_validation_label_list]), axes=[0, 2, 3, 1]) / 255
#     print(fold_validation_input.shape, fold_validation_label.shape)
#
#     # Train Each Fold
#
#     history = model.fit(fold_train_input,
#                         fold_train_label,
#                         # class_weight=[0.6, 1],
#                         validation_data=(fold_validation_input, fold_validation_label),
#                         epochs=30,
#                         batch_size=16,
#                         verbose=2)


train_loaded = np.load('./Dataset/trainset_256.npz')
test_loaded = np.load('./Dataset/testset_256.npz')

train_input = train_loaded['dataset']
train_label = train_loaded['label']

validation_input = test_loaded['dataset']
validation_label = test_loaded['label']

# options = 'Unet-256_256_32-RMSprop_1e-6-diceloss-elu-he-batch_12'
#options = 'Unet-256_256_32-Nadam_1e-4-diceloss-relu-he-batch_32'
options = 'DeeplabV3-256_256_8-Nadam_1e-4-diceloss-relu-he-batch_8'
filepath = './models/'+options+'-epoch.{epoch:02d}.hdf5'
ckpt_callback = ModelCheckpoint(filepath=filepath)


# model = Unet(input_shape=[256, 256, 1], channel_size=32)
model = DeeplabV3(input_shape=[256, 256, 1], channel_size=8)

model.summary()
##multi-gpu
#model = multi_gpu_model(model, gpus=2, cpu_merge=False)
#model.summary()

model.compile(optimizer=Nadam(lr=1e-4), loss=dice_coef_loss, metrics=[iou_coef, dice_coef])


history = model.fit(train_input,
                    train_label,
                    validation_data=(validation_input, validation_label),
                    epochs=10,
                    batch_size=8,
                    verbose=2,
                    callbacks=[ckpt_callback])

epochs = range(1, 50 + 1)

import matplotlib.pyplot as plt

val_loss = history.history['val_loss']
val_iou = history.history['val_iou_coef']
val_dice = history.history['val_dice_coef']

plot = plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plot.title('Validation Loss')
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.legend()
plot.savefig('D:/loss.png', dpi=600)

plot = plt.plot(epochs, val_iou, 'b', label='Validation IoU')
plot.title('Validation IoU')
plot.xlabel('Epochs')
plot.ylabel('IoU')
plot.legend()
plot.savefig('D:/IoU.png', dpi=600)

plot = plt.plot(epochs, val_dice, 'b', label='Validation Dice')
plot.title('Validation Dice')
plot.xlabel('Epochs')
plot.ylabel('dice')
plot.legend()
plot.savefig('D:/Dice.png', dpi=600)

"""
Unet(input_shape=[256, 256, 1], channel_size=32)
optimizer=Nadam(lr=1e-4), loss=dice_coef_loss, metrics=[iou_coef, dice_coef]
relu
he_normal
epoch 50
batch size 32
"""
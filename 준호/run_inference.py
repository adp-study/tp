from time import time
import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from tensorflow.keras import models

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def Unet(input_shape=[128, 128, 1], channel_size=8):
    n1_Input_o1 = Input(shape=input_shape, batch_shape=None, name='n1_Input')

    n2_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n1_Input_o1)
    n2_contract_n2_BatchNormalization_o1 = BatchNormalization()(n2_contract_n1_Conv2D_o1)
    n2_contract_n3_Activation_o1 = Activation(activation='relu')(n2_contract_n2_BatchNormalization_o1)
    n2_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n2_contract_n3_Activation_o1)
    n2_contract_n5_BatchNormalization_o1 = BatchNormalization()(n2_contract_n4_Conv2D_o1)
    n2_contract_n6_Activation_o1 = Activation(activation='relu')(n2_contract_n5_BatchNormalization_o1)

    n5_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n2_contract_n6_Activation_o1)

    n6_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n5_MaxPooling2D_o1)
    n6_contract_n2_BatchNormalization_o1 = BatchNormalization()(n6_contract_n1_Conv2D_o1)
    n6_contract_n3_Activation_o1 = Activation(activation='relu')(n6_contract_n2_BatchNormalization_o1)
    n6_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n6_contract_n3_Activation_o1)
    n6_contract_n5_BatchNormalization_o1 = BatchNormalization()(n6_contract_n4_Conv2D_o1)
    n6_contract_n6_Activation_o1 = Activation(activation='relu')(n6_contract_n5_BatchNormalization_o1)

    n7_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n6_contract_n6_Activation_o1)

    n8_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n7_MaxPooling2D_o1)
    n8_contract_n2_BatchNormalization_o1 = BatchNormalization()(n8_contract_n1_Conv2D_o1)
    n8_contract_n3_Activation_o1 = Activation(activation='relu')(n8_contract_n2_BatchNormalization_o1)
    n8_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                      kernel_initializer='he_normal')(n8_contract_n3_Activation_o1)
    n8_contract_n5_BatchNormalization_o1 = BatchNormalization()(n8_contract_n4_Conv2D_o1)
    n8_contract_n6_Activation_o1 = Activation(activation='relu')(n8_contract_n5_BatchNormalization_o1)

    n9_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n8_contract_n6_Activation_o1)

    n10_contract_n1_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                       kernel_initializer='he_normal')(n9_MaxPooling2D_o1)
    n10_contract_n2_BatchNormalization_o1 = BatchNormalization()(n10_contract_n1_Conv2D_o1)
    n10_contract_n3_Activation_o1 = Activation(activation='relu')(n10_contract_n2_BatchNormalization_o1)
    n10_contract_n4_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                       kernel_initializer='he_normal')(n10_contract_n3_Activation_o1)
    n10_contract_n5_BatchNormalization_o1 = BatchNormalization()(n10_contract_n4_Conv2D_o1)
    n10_contract_n6_Activation_o1 = Activation(activation='relu')(n10_contract_n5_BatchNormalization_o1)

    n11_MaxPooling2D_o1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(n10_contract_n6_Activation_o1)

    n3_bottle_neck_n1_Conv2D_o1 = Conv2D(filters=channel_size * 16, kernel_size=[3, 3], padding='same',
                                         kernel_initializer='he_normal')(n11_MaxPooling2D_o1)
    n3_bottle_neck_n2_BatchNormalization_o1 = BatchNormalization()(n3_bottle_neck_n1_Conv2D_o1)
    n3_bottle_neck_n3_Activation_o1 = Activation(activation='relu')(n3_bottle_neck_n2_BatchNormalization_o1)
    n3_bottle_neck_n4_Conv2D_o1 = Conv2D(filters=channel_size * 16, kernel_size=[3, 3], padding='same',
                                         kernel_initializer='he_normal')(n3_bottle_neck_n3_Activation_o1)
    n3_bottle_neck_n5_BatchNormalization_o1 = BatchNormalization()(n3_bottle_neck_n4_Conv2D_o1)
    n3_bottle_neck_n6_Activation_o1 = Activation(activation='relu')(n3_bottle_neck_n5_BatchNormalization_o1)

    n12_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size * 8, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(
        n3_bottle_neck_n6_Activation_o1)

    n4_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n12_Conv2DTranspose_o1, n10_contract_n6_Activation_o1])
    n4_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                    kernel_initializer='he_normal')(n4_expand_n1_Concatenate_o1)
    n4_expand_n3_BatchNormalization_o1 = BatchNormalization()(n4_expand_n2_Conv2D_o1)
    n4_expand_n4_Activation_o1 = Activation(activation='relu')(n4_expand_n3_BatchNormalization_o1)
    n4_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size * 8, kernel_size=[3, 3], padding='same',
                                    kernel_initializer='he_normal')(n4_expand_n4_Activation_o1)
    n4_expand_n6_BatchNormalization_o1 = BatchNormalization()(n4_expand_n5_Conv2D_o1)
    n4_expand_n7_Activation_o1 = Activation(activation='relu')(n4_expand_n6_BatchNormalization_o1)

    n13_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size * 4, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(
        n4_expand_n7_Activation_o1)

    n14_expand_n1_Concatenate_o1 = Concatenate(axis=-1, name='n14_expand_n1_Concatenate')(
        [n13_Conv2DTranspose_o1, n8_contract_n6_Activation_o1])
    n14_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n14_expand_n1_Concatenate_o1)
    n14_expand_n3_BatchNormalization_o1 = BatchNormalization()(n14_expand_n2_Conv2D_o1)
    n14_expand_n4_Activation_o1 = Activation(activation='relu')(n14_expand_n3_BatchNormalization_o1)
    n14_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size * 4, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n14_expand_n4_Activation_o1)
    n14_expand_n6_BatchNormalization_o1 = BatchNormalization()(n14_expand_n5_Conv2D_o1)
    n14_expand_n7_Activation_o1 = Activation(activation='relu')(n14_expand_n6_BatchNormalization_o1)

    n15_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size * 2, kernel_size=[2, 2], strides=[2, 2],
                                             padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(
        n14_expand_n7_Activation_o1)

    n16_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n15_Conv2DTranspose_o1, n6_contract_n6_Activation_o1])
    n16_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n16_expand_n1_Concatenate_o1)
    n16_expand_n3_BatchNormalization_o1 = BatchNormalization()(n16_expand_n2_Conv2D_o1)
    n16_expand_n4_Activation_o1 = Activation(activation='relu')(n16_expand_n3_BatchNormalization_o1)
    n16_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size * 2, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n16_expand_n4_Activation_o1)
    n16_expand_n6_BatchNormalization_o1 = BatchNormalization()(n16_expand_n5_Conv2D_o1)
    n16_expand_n7_Activation_o1 = Activation(activation='relu')(n16_expand_n6_BatchNormalization_o1)

    n17_Conv2DTranspose_o1 = Conv2DTranspose(filters=channel_size, kernel_size=[2, 2], strides=[2, 2], padding='valid',
                                             output_padding=[0, 0], dilation_rate=[1, 1])(n16_expand_n7_Activation_o1)

    n18_expand_n1_Concatenate_o1 = Concatenate(axis=-1)([n17_Conv2DTranspose_o1, n2_contract_n6_Activation_o1])
    n18_expand_n2_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n18_expand_n1_Concatenate_o1)
    n18_expand_n3_BatchNormalization_o1 = BatchNormalization()(n18_expand_n2_Conv2D_o1)
    n18_expand_n4_Activation_o1 = Activation(activation='relu')(n18_expand_n3_BatchNormalization_o1)
    n18_expand_n5_Conv2D_o1 = Conv2D(filters=channel_size, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(n18_expand_n4_Activation_o1)
    n18_expand_n6_BatchNormalization_o1 = BatchNormalization()(n18_expand_n5_Conv2D_o1)
    n18_expand_n7_Activation_o1 = Activation(activation='relu')(n18_expand_n6_BatchNormalization_o1)

    n19_Conv2D_o1 = Conv2D(filters=2, kernel_size=[1, 1], strides=[1, 1], padding='same', activation='softmax',
                           kernel_initializer='he_normal')(n18_expand_n7_Activation_o1)

    model = Model(inputs=n1_Input_o1, outputs=[n19_Conv2D_o1])
    return model


st = time()
best_model_path = './models/Unet-256_256_32-Nadam_1e-4-diceloss-relu-he-batch_32-epoch.50.hdf5'
test_loaded = np.load('./Dataset/testset_256.npz')

# best_model_path = './models/Unet-256_256_32-Nadam_1e-4-diceloss-relu-he-batch_32-epoch.01.hdf5'
# test_loaded = np.load('./Dataset/testset_256.npz')

#print(validation_input.shape)
#validation_input = np.expand_dims(test_loaded['dataset'][0,:,:,:], axis=0)

validation_input = test_loaded['dataset']
validation_label = test_loaded['label']

print(validation_input.shape, validation_label.shape)


## inference run function
def run_inference(array):
    best_model_path = './models/Unet-192_192_32-Nadam_1e-4-diceloss-relu-he-batch_32-epoch.40.hdf5'
    validation_input = np.squeeze(array, axis=0) # (576, 576)
    validation_input = np.squeeze(validation_input, axis=-1) # (576, 576)
    org_shape = np.squeeze(validation_input.shape)
    validation_input = cv2.resize(src=validation_input, dsize=(192, 192), interpolation=cv2.INTER_AREA)
    validation_input = (validation_input - np.min(validation_input))/np.max(validation_input) * 255
    validation_input = np.expand_dims(validation_input, axis=0)
    validation_input = np.expand_dims(validation_input, axis=-1)

    new_model = models.load_model(best_model_path, custom_objects={'iou_coef':iou_coef, 'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})
    new_model.summary()
    pred = new_model.predict(validation_input)
    pred_image = pred[:, :, :, 1] * 255
    pred_image = np.squeeze(pred_image, axis=0) # (576, 576)
    pred_image = cv2.resize(src=pred_image, dsize=(org_shape[0], org_shape[1]), interpolation=cv2.INTER_AREA)
    return np.uint8(pred_image)


## run for each file
## And save ground truth file, prediction file, overlapped with original file saving as image file using CV2
for slice in range(validation_input.shape[0]):
    vinput = np.expand_dims(validation_input[slice,:,:,:], axis=0)
    n_ar = (vinput - np.min(vinput)) / np.max(vinput) * 255
    array_image = np.expand_dims(vinput, axis=-1)
    pred = run_inference(array_image)
    gt_img = validation_label[slice, :, :, 1] * 255

    gt_sp = './res/gt1/' + str(slice) + '.png'
    pr_sp = './res/pr1/' + str(slice) + '.png'
    ov_sp = './res/ov1/' + str(slice) + '.png'
    if not os.path.exists(os.path.dirname(gt_sp)):
        os.makedirs(os.path.dirname(gt_sp))
    if not os.path.exists(os.path.dirname(pr_sp)):
        os.makedirs(os.path.dirname(pr_sp))
    if not os.path.exists(os.path.dirname(ov_sp)):
        os.makedirs(os.path.dirname(ov_sp))

    cv2.imwrite(gt_sp, gt_img)
    cv2.imwrite(pr_sp, pred)

    label = pred
    test_image = np.squeeze(vinput)
    pred_image = label
    pred_image = np.expand_dims(pred_image, axis=0)
    pred_image = np.expand_dims(pred_image, axis=3)
    G = np.zeros([1, 256, 256, 1])
    B = np.zeros([1, 256, 256, 1])
    R = pred_image
    pred_image = np.concatenate((B, G, R), axis=3)
    pred_image = np.squeeze(pred_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.expand_dims(test_image, axis=3)
    tR = test_image
    tG = test_image
    tB = test_image
    test_image = np.concatenate((tB, tG, tR), axis=3)
    test_image = np.squeeze(test_image)
    test_image = test_image.astype(float)
    w = 40
    p = 0.0001
    result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0) * 255
    cv2.imwrite(ov_sp, result)
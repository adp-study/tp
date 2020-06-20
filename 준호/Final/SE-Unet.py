import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization, Add, Reshape, Multiply, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
import tensorflow.keras.backend as K
import tensorflow as tf
import os



os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='selu'):
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    if activation is None:
        x = x
    else:
        x = Activation(activation)(x)
    return x


def SE_block(input_tensor, reduction_ratio=16):
    ch_input = K.int_shape(input_tensor)[-1]
    ch_reduced = ch_input // reduction_ratio
    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor)
    # Excitation
    x = Dense(ch_reduced, kernel_initializer='he_normal', activation='selu', use_bias=False)(x)
    x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x)
    x = Reshape((1, 1, ch_input))(x)
    x = Multiply()([input_tensor, x])
    return x


def SE_residual_block(input_tensor, filter_sizes, strides=1, reduction_ratio=16):
    filter_1, filter_2, filter_3 = filter_sizes
    x = conv2d_bn(input_tensor, filter_1, (1, 1), strides=strides)
    x = conv2d_bn(x, filter_2, (3, 3))
    x = conv2d_bn(x, filter_3, (1, 1), activation=None)
    x = SE_block(x, reduction_ratio)
    projected_input = conv2d_bn(input_tensor, filter_3, (1, 1), strides=strides, activation=None) if \
    K.int_shape(input_tensor)[-1] != filter_3 else input_tensor
    shortcut = Add()([projected_input, x])
    shortcut = Activation(activation='selu')(shortcut)
    return shortcut


def stage_downsample_block(input_tensor, filter_sizes, blocks, reduction_ratio=16, stage=''):
    strides = 2 if stage != '2' else 1
    x = SE_residual_block(input_tensor, filter_sizes, strides, reduction_ratio)  # projection layer
    for i in range(blocks - 1):
        x = SE_residual_block(x, filter_sizes, reduction_ratio=reduction_ratio)
    return x


def SE_ResNet50_downsampling(model_input, channel_size):
    stage_1 = stage_downsample_block(model_input, [channel_size, channel_size, channel_size * 2], 3, reduction_ratio=8, stage='2')
    stage_2 = stage_downsample_block(stage_1, [channel_size, channel_size, channel_size * 4], 4, reduction_ratio=8, stage='3')  # (28, 28, 512)
    stage_3 = stage_downsample_block(stage_2, [channel_size, channel_size, channel_size * 8], 6, reduction_ratio=8, stage='4')  # (14, 14, 1024)
    stage_4 = stage_downsample_block(stage_3, [channel_size, channel_size, channel_size * 16], 3, reduction_ratio=8, stage='5')  # (7, 7, 2048)
    stage_4_pool = MaxPooling2D((3, 3), strides=2, padding='same')(stage_4)
    return [stage_4_pool, stage_4, stage_3, stage_2, stage_1], channel_size*16


def stage_upsample_block(input_tensor, channel_size, upsample_target, filter_sizes=(3,3), stage=''):
    x = Conv2DTranspose(filters=channel_size // 2, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(input_tensor)
    concat = Concatenate(axis=-1)([x, upsample_target])
    x = conv2d_bn(concat, channel_size, filter_sizes, strides=1, padding='same')
    x = conv2d_bn(x, channel_size, filter_sizes, strides=1, padding='same')
    shortcut = Add()([concat, x])
    return shortcut


def Upsampling(layers, channel_size):
    x = layers[0]
    for i in range(len(layers)-1):
        channel_size = channel_size // 2
        x = conv2d_bn(x, channel_size, (3, 3), strides=1, padding='same')
        x = conv2d_bn(x, channel_size, (3, 3), strides=1, padding='same')
        x = Conv2DTranspose(filters=channel_size, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(x)
        x = Concatenate(axis=-1)([x, layers[i+1]])
    return x


def SE_Unet(input_shape, channel_size=16):
    model_input = Input(shape=input_shape, batch_shape=None, name='Input')
    downsamples, last_channel_size = SE_ResNet50_downsampling(model_input, channel_size)
    upsampled = Upsampling(downsamples, last_channel_size)
    model_output = conv2d_bn(upsampled, channel_size*4, kernel_size=(3,3), strides=1, padding='same')
    model_output = conv2d_bn(model_output, channel_size, kernel_size=(3, 3), strides=1, padding='same')
    model_output = Conv2D(2, [1, 1], kernel_initializer='he_normal', padding='same', strides=[1, 1], activation='softmax')(model_output)
    model = Model(inputs=model_input, outputs=model_output, name='SE-ResNet50-Unet')
    return model


# Dice score
def dice_coef(y_true, y_pred):
    # dsc = K.sum(y_pred[y_true == 255]) * 2.0 / (K.sum(y_pred) + K.sum(y_true) + K.epsilon())
    # print(y_true.shape, y_pred.shape)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_np(y_true, y_pred):
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    return (2. * intersection + 1e-5) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)



# train_loaded = np.load('/home/bjh/home/bjh/PythonProjects/NonDeepphi/HW/Dataset/trainset_256.npz')
# test_loaded = np.load('/home/bjh/home/bjh/PythonProjects/NonDeepphi/HW/Dataset/testset_256.npz')
# train_loaded = np.load('E:/PythonProjects/NonDeepphi/HW/Dataset/trainset_256.npz')
# test_loaded = np.load('E:/PythonProjects/NonDeepphi/HW/Dataset/testset_256.npz')
train_loaded = np.load('./trainset_256.npz')
test_loaded = np.load('./testset_256.npz')

train_input = train_loaded['dataset']
train_label = train_loaded['label']

validation_input = test_loaded['dataset']
validation_label = test_loaded['label']



class CallBacks(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        super(CallBacks, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        y_pred_val = self.model.predict(self.x_val)

        trdscl = []
        vldscl = []

        for s in range(self.y.shape[0]):
            predimg = y_pred[s, :, :, 1]
            gtimg = self.y[s, :, :, 1]
            dsc = dice_coef_np(gtimg, predimg)
            trdscl.append(dsc)
        tr_dsc = np.mean(trdscl)

        for s in range(self.y_val.shape[0]):
            predimg = y_pred_val[s, :, :, 1]
            gtimg = self.y_val[s, :, :, 1]
            dsc = dice_coef_np(gtimg, predimg)
            vldscl.append(dsc)
        vl_dsc = np.mean(vldscl)

        ep = epoch+1

        print('Epoch {} DSC for target class : Train DSC {}, Validation DSC {}'.format(ep, tr_dsc, vl_dsc))
        if vl_dsc >= 0.9:
            spath = "./models/final_SE-Res-Unet-epoch-{}-val-DSC-{}-relu.hdf5".format(ep, vl_dsc)
            print(spath)
            self.model.save(spath)
        return


model = SE_Unet(input_shape=(256, 256, 1), channel_size=8)
model.summary()
model.compile(optimizer=Nadam(lr=1e-3), loss=dice_coef_loss)
callbacks = CallBacks(training_data=(train_input, train_label), validation_data=(validation_input, validation_label))
history = model.fit(train_input,
                    train_label,
                    validation_data=(validation_input, validation_label),
                    epochs=100,
                    batch_size=4,
                    verbose=2,
                    shuffle=True,
                    callbacks=[callbacks])#, callbacks=[ckpt_callback])

import pickle
with open('./trainres_dict_gpu0_selu.pkl', 'wb') as pkl:
    pickle.dump(history.history, pkl)

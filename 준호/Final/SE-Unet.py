import pickle
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization, Add, Reshape, Multiply, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
import os


## Force to environment to use GPU 1 only
## IF want to use CPU only, put '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

## For WINDOWS 10
## OS Windows, GPU MEMORY Allocation is blocked by firewall, controlled by os.
## This code will be unlock GPU memory Allocation.
## IF erase this, GPU Memory Error will be occured.
## For Linux, This code SHOULD BE ELIMINATED
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    """
    Basic convolution block. includes Conv2D-Activation-Batchnormalization 
    """
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    if activation is None:
        x = x
    else:
        x = Activation(activation)(x)
    return x


def SE_block(input_tensor, reduction_ratio=16):
    """
    Squeeze-Excitation block. Use global average pooling for squeeze feature, recalibration by channel reduced dense layer, reconstruction by channel recovered dense layer.
    """
    ch_input = K.int_shape(input_tensor)[-1]
    ch_reduced = ch_input // reduction_ratio
    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor)
    # Excitation
    x = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(x)
    x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x)
    x = Reshape((1, 1, ch_input))(x)
    x = Multiply()([input_tensor, x])
    return x


def SE_residual_block(input_tensor, filter_sizes, strides=1, reduction_ratio=16):
    """
    Residual block with SE block. bottleneck structure is in this block(by using 1x1 conv layer)
    """
    filter_1, filter_2, filter_3 = filter_sizes
    x = conv2d_bn(input_tensor, filter_1, (1, 1), strides=strides)
    x = conv2d_bn(x, filter_2, (3, 3))
    x = conv2d_bn(x, filter_3, (1, 1), activation=None)
    x = SE_block(x, reduction_ratio)
    projected_input = conv2d_bn(input_tensor, filter_3, (1, 1), strides=strides, activation=None) if \
    K.int_shape(input_tensor)[-1] != filter_3 else input_tensor
    shortcut = Add()([projected_input, x])
    shortcut = Activation(activation='relu')(shortcut)
    return shortcut


def stage_downsample_block(input_tensor, filter_sizes, blocks, reduction_ratio=16, stage=''):
    """
    Downsampling block for Res-SE-Unet. returns downsampled result.
    Max pooling is not used for downsizing, stride is used.
    Because max pooling erases some feature information when doing pooling, this is critical for segmentation.
    """
    strides = 2 if stage != '2' else 1
    x = SE_residual_block(input_tensor, filter_sizes, strides, reduction_ratio)  # projection layer
    for i in range(blocks - 1):
        x = SE_residual_block(x, filter_sizes, reduction_ratio=reduction_ratio)
    return x


def SE_ResNet50_downsampling(model_input, channel_size):
    """
    Residual SE block with 50 Conv layer.
    """
    stage_1 = stage_downsample_block(model_input, [channel_size, channel_size, channel_size * 2], 3, reduction_ratio=8, stage='2')
    stage_2 = stage_downsample_block(stage_1, [channel_size, channel_size, channel_size * 4], 4, reduction_ratio=8, stage='3')  # (28, 28, 512)
    stage_3 = stage_downsample_block(stage_2, [channel_size, channel_size, channel_size * 8], 6, reduction_ratio=8, stage='4')  # (14, 14, 1024)
    stage_4 = stage_downsample_block(stage_3, [channel_size, channel_size, channel_size * 16], 3, reduction_ratio=8, stage='5')  # (7, 7, 2048)
    stage_4_pool = MaxPooling2D((3, 3), strides=2, padding='same')(stage_4)
    return [stage_4_pool, stage_4, stage_3, stage_2, stage_1], channel_size*16


def Upsampling(layers, channel_size):
    """
    Upsampling block for Res-SE-Unet.
    Transpose Convolution is used for Upsampling method.
    Upsampling by Image processing might be work.
    """
    x = layers[0]
    for i in range(len(layers)-1):
        channel_size = channel_size // 2
        x = conv2d_bn(x, channel_size, (3, 3), strides=1, padding='same')
        x = conv2d_bn(x, channel_size, (3, 3), strides=1, padding='same')
        x = Conv2DTranspose(filters=channel_size, kernel_size=[2, 2], strides=[2, 2], padding='valid', output_padding=[0, 0], dilation_rate=[1, 1])(x)
        x = Concatenate(axis=-1)([x, layers[i+1]])
    return x


def SE_Unet(input_shape, channel_size=16):
    """
    Total Architecture code for Res-SE-Unet
    returns keras Model.
    """
    model_input = Input(shape=input_shape, batch_shape=None, name='Input')
    downsamples, last_channel_size = SE_ResNet50_downsampling(model_input, channel_size)
    upsampled = Upsampling(downsamples, last_channel_size)
    model_output = conv2d_bn(upsampled, channel_size*4, kernel_size=(3,3), strides=1, padding='same')
    model_output = conv2d_bn(model_output, channel_size, kernel_size=(3, 3), strides=1, padding='same')
    model_output = Conv2D(2, [1, 1], kernel_initializer='he_normal', padding='same', strides=[1, 1], activation='softmax')(model_output)
    model = Model(inputs=model_input, outputs=model_output, name='SE-ResNet50-Unet')
    return model


def dice_coef(y_true, y_pred):
    """
    dice coefficient for keras backend
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    """
    custom loss function(dice loss) for keras backend
    """
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_np(y_true, y_pred):
    """
    calculate DSC for predict result.
    If you want to calculate DSC of target class only, slice last channel(1) and put into function.
    Segmentation result of DSC is easily corrupted by background pixels
    (many tasks of segmentation have more pixels of background than target pixels.
     so when calcuate DSC, if put whole prediction result, DSC will be increased by background pixels)
    """
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    return (2. * intersection + 1e-5) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)


# Custom activation functions for keras
class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(x):
    return x * K.tanh(K.softplus(x))

get_custom_objects().update({'mish': Mish(mish)})


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'

def swish(x):
    return x * K.sigmoid(x)

get_custom_objects().update({'Swish': Swish(swish)})


## Dataset loading
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
    """
    Custom Callback class for keras.
    used for model saving by some options, calculate some custom measurements.
    """
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        super(CallBacks, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """
        works when each epoch ends
        print DSC after epoch finished, saving models for some condition
        """
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


if __name__ is '__main__':
    model = SE_Unet(input_shape=(256, 256, 1), channel_size=8)
    model.summary()
    model.compile(optimizer=Nadam(lr=1e-3), loss=dice_coef_loss)
    callbacks = CallBacks(training_data=(train_input, train_label), validation_data=(validation_input, validation_label))

    # train run
    history = model.fit(train_input,
                        train_label,
                        validation_data=(validation_input, validation_label),
                        epochs=100,
                        batch_size=4,
                        verbose=2,
                        shuffle=True,
                        callbacks=[callbacks])

    # save train history for python pickle.
    with open('./trainres_dict_gpu0_selu.pkl', 'wb') as pkl:
        pickle.dump(history.history, pkl)

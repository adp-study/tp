import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ## IF GPU RUN, ELIMINATE THIS CODE
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


# Dice score
def dice_coef_np(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    # print(np.sum(y_true_f), np.sum(y_pred_f), intersection)
    # print(2. * intersection + 1e-5)
    # print((np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5))
    return (2. * intersection + 1e-5) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)


def run_inference_for_dsc(array, label, new_model):
    validation_input = np.squeeze(array, axis=0) # (576, 576)
    validation_input = np.squeeze(validation_input, axis=-1) # (576, 576)
    # org_shape = np.squeeze(validation_input.shape)
    # validation_input = cv2.resize(src=validation_input, dsize=(192, 192), interpolation=cv2.INTER_AREA)
    validation_input = (validation_input - np.min(validation_input))/np.max(validation_input) * 255
    validation_input = np.expand_dims(validation_input, axis=0)
    validation_input = np.expand_dims(validation_input, axis=-1)
    pred = new_model.predict(validation_input)
    print(pred.shape, label.shape)
    label = label[0,:,:,1]
    pred = pred[0,:,:,1]
    print(pred.shape, label.shape)
    dsc = dice_coef_np(y_true=label, y_pred=pred)
    return dsc#, label, pred

###################
test_loaded = np.load('./testset_256.npz')

validation_input = test_loaded['dataset']
validation_label = test_loaded['label']
# best_model_path = './SE-Res-Unet-epoch-5-val-DSC-0.8444443215445779.hdf5'
# best_model_path = './SE-Res-Unet-epoch-10-val-DSC-0.8744654542506171.hdf5'
# best_model_path = './SE-Res-Unet-epoch-30-val-DSC-0.8896102045079533.hdf5'
# best_model_path = './SE-Res-Unet-epoch-50-val-DSC-0.8906427648871281.hdf5'
# best_model_path = './SE-Res-Unet-epoch-100-val-DSC-0.8953325372780416.hdf5'
best_model_path = './SE-Res-Unet-epoch-116-val-DSC-0.9062188331879039-gpu6.hdf5'

new_model = models.load_model(best_model_path, custom_objects={'iou_coef': iou_coef,
                                                               'dice_coef': dice_coef,
                                                               'dice_coef_loss': dice_coef_loss})
new_model.summary()

preds = new_model.predict(validation_input)
print(np.max(validation_input), validation_input.shape)
print(np.max(preds), preds.shape, np.max(validation_label), validation_label.shape)
wdsc = dice_coef_np(validation_label[:,:,:,1], preds[:,:,:,1])
print(wdsc)
#255 (1456, 256, 256, 1)
#1.0 (1456, 256, 256, 2) 1.0 (1456, 256, 256, 2)
dsclist = []
for slice in range(validation_input.shape[0]):
    inputimg = validation_input[slice,:,:,0]
    predimg = preds[slice,:,:,1]
    gtimg = validation_label[slice,:,:,1]
    print(inputimg.shape, predimg.shape, gtimg.shape)
    dsc = dice_coef_np(gtimg, predimg)
    print(dsc)
    dsclist.append(dsc)

    inputsp = './res_epoch116/input/' + str(slice) + '.png'
    gtsp = './res_epoch116/gt/' + str(slice) + '.png'
    predsp = './res_epoch116/pred/' + str(slice) + '.png'
    inputpredovsp = './res_epoch116/input-gt/' + str(slice) + '.png'
    gtpredovsp = './res_epoch116/input-pred/' + str(slice) + '.png'
    if not os.path.exists(os.path.dirname(inputsp)):
        os.makedirs(os.path.dirname(inputsp))
    if not os.path.exists(os.path.dirname(gtsp)):
        os.makedirs(os.path.dirname(gtsp))
    if not os.path.exists(os.path.dirname(predsp)):
        os.makedirs(os.path.dirname(predsp))
    if not os.path.exists(os.path.dirname(inputpredovsp)):
        os.makedirs(os.path.dirname(inputpredovsp))
    if not os.path.exists(os.path.dirname(gtpredovsp)):
        os.makedirs(os.path.dirname(gtpredovsp))

    test_image = inputimg
    pred_image = gtimg
    print(pred_image.shape)
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
    # print(pred_image.shape, test_image.shape)
    result = cv2.addWeighted(pred_image*255, float(100 - w) * p, test_image, float(w) * p, 0) * 255
    # test_image = np.squeeze(vinput)

    test_image = inputimg
    pred_image = predimg
    pred_image = np.expand_dims(pred_image, axis=0)
    pred_image = np.expand_dims(pred_image, axis=3)
    G = pred_image
    B = np.zeros([1, 256, 256, 1])
    R = np.zeros([1, 256, 256, 1])
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
    # print(pred_image.shape, test_image.shape)
    gtpr_img = cv2.addWeighted(pred_image*255, float(100 - w) * p, test_image, float(w) * p, 0) * 255

    cv2.imwrite(inputsp, inputimg)
    cv2.imwrite(gtsp, gtimg*255)
    cv2.imwrite(predsp, predimg*255)
    cv2.imwrite(inputpredovsp, result)
    cv2.imwrite(gtpredovsp, gtpr_img)

print(np.mean(dsclist))

# for slice in range(validation_input.shape[0]):
# # for slice in [0, 3, 4, 5, 8, 9, 11, 14, 15, 17, 18, 22, 23, 25, 31, 32, 34, 35, 37, 38, 40, 41, 43, 46, 47, 49, 51, 53, 56, 59, 60, 61, 63, 64, 66, 67, 69, 70, 71, 72, 75, 76, 79, 80, 81, 85, 86, 87, 88, 89, 90, 92, 94, 95, 97, 99, 100, 102, 105, 106, 107, 108, 110, 111, 112, 114, 115, 116, 117, 118, 120, 121, 124, 127, 128, 129, 131, 132, 133, 135, 136, 138, 139, 140, 143, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161]:
#     vinput = np.expand_dims(validation_input[slice, :, :, :], axis=0)
#     vlabel = np.expand_dims(validation_label[slice, :, :, :], axis=0)
#     # print(vinput.shape, vlabel.shape)
#     n_ar = (vinput - np.min(vinput)) / np.max(vinput) * 255
#     array_image = vinput #np.expand_dims(vinput, axis=-1)
#     dsc, label, pred = run_inference_for_dsc(array_image, vlabel, new_model)
#     print('DSC', dsc)
#     array_image = np.expand_dims(n_ar, axis=-1)
#     gt_img = label
#     print(np.max(array_image), np.max(vlabel), np.max(gt_img), np.max(label), np.max(pred))
#
#     inputsp = './res_epoch5/input/'+str(slice)+'.png'
#     gtsp = './res_epoch5/gt/'+str(slice)+'.png'
#     predsp = './res_epoch5/pred/'+str(slice)+'.png'
#     inputpredovsp = './res_epoch5/input-pred/'+str(slice)+'.png'
#     gtpredovsp = './res_epoch5/gt-pred/'+str(slice)+'.png'
#     concatsp = './res_epoch5/concat/' + str(slice) + '.png'
#
#     if not os.path.exists(os.path.dirname(inputsp)):
#         os.makedirs(os.path.dirname(inputsp))
#     if not os.path.exists(os.path.dirname(gtsp)):
#         os.makedirs(os.path.dirname(gtsp))
#     if not os.path.exists(os.path.dirname(predsp)):
#         os.makedirs(os.path.dirname(predsp))
#     if not os.path.exists(os.path.dirname(inputpredovsp)):
#         os.makedirs(os.path.dirname(inputpredovsp))
#     if not os.path.exists(os.path.dirname(gtpredovsp)):
#         os.makedirs(os.path.dirname(gtpredovsp))
#     if not os.path.exists(os.path.dirname(concatsp)):
#         os.makedirs(os.path.dirname(concatsp))
#
#     test_image = array_image
#
#     pred_image = label
#     print(pred_image.shape)
#     pred_image = np.expand_dims(pred_image, axis=0)
#     pred_image = np.expand_dims(pred_image, axis=3)
#     G = np.zeros([1, 256, 256, 1])
#     B = np.zeros([1, 256, 256, 1])
#     R = pred_image
#
#     pred_image = np.concatenate((B, G, R), axis=3)
#     pred_image = np.squeeze(pred_image)
#
#     test_image = np.expand_dims(test_image, axis=0)
#     test_image = np.expand_dims(test_image, axis=3)
#     tR = test_image
#     tG = test_image
#     tB = test_image
#     test_image = np.concatenate((tB, tG, tR), axis=3)
#     test_image = np.squeeze(test_image)
#     test_image = test_image.astype(float)
#     w = 40
#     p = 0.0001
#     # print(pred_image.shape, test_image.shape)
#     result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0) * 255
#
#         # test_image = np.squeeze(vinput)
#     test_image = array_image
#     pred_image = label
#     pred_image = np.expand_dims(pred_image, axis=0)
#     pred_image = np.expand_dims(pred_image, axis=3)
#     G = pred_image
#     B = np.zeros([1, 256, 256, 1])
#     R = np.zeros([1, 256, 256, 1])
#
#     pred_image = np.concatenate((B, G, R), axis=3)
#     pred_image = np.squeeze(pred_image)
#
#     test_image = np.expand_dims(test_image, axis=0)
#     test_image = np.expand_dims(test_image, axis=3)
#     tR = test_image
#     tG = test_image
#     tB = test_image
#     test_image = np.concatenate((tB, tG, tR), axis=3)
#     test_image = np.squeeze(test_image)
#     test_image = test_image.astype(float)
#     w = 40
#     p = 0.0001
#     # print(pred_image.shape, test_image.shape)
#     gtpr_img = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0) * 255
#
#     input_img = np.transpose(np.array([array_image, array_image, array_image]), axes=[1, 2, 0])
#
#     cc_img = np.hstack([input_img, gtpr_img, result])
#     #
#     cv2.imwrite(inputsp, array_image)
#     cv2.imwrite(gtsp, gt_img)
#     cv2.imwrite(predsp, pred)
#     cv2.imwrite(inputpredovsp, result)
#     cv2.imwrite(gtpredovsp, gtpr_img)
#     cv2.imwrite(concatsp, cc_img)

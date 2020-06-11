import cv2
import nibabel
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom as pdcm
import pydicom.uid
import warnings
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom


# def resample_img(img, spacing, new_spacing=[1, 1, 1], order=2):
#     if len(img.shape) == 3:
#         new_shape = np.round(img.shape * spacing / new_spacing)
#         resize_factor = new_shape / img.shape
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             img = zoom(img, resize_factor, mode='nearest', order=order)
#         return img
#     elif len(img.shape) == 4:
#         n = img.shape[-1]
#         new_img = []
#         for i in range(n):
#             img_slice = img[:, :, :, i]
#             new_slice = resample_img(img_slice, spacing, new_spacing)
#             new_img.append(new_slice)
#         new_img = np.transpose(np.array(new_img), [1, 2, 3, 0])
#         return new_img
#     else:
#         raise ValueError('wrong shape')


def resample(img, org_spacing, new_spacing=[1, 1]):
    resize_factors = []
    for _o, _n in zip(org_spacing, new_spacing):
        resize_factors.append(_o * _n)
    resize_factors.append(dcm_thickness)
    return zoom(img, resize_factors, mode='wrap', order=2)


for r_dpath in [os.path.join('E:/Dataset/Disc/Dataset', d) for d in os.listdir('E:/Dataset/Disc/Dataset')]:
    r_lpath = r_dpath.replace('Disc/Dataset', 'Disc/Label')
    try:
        _dcmimgs = [pdcm.dcmread(os.path.join(r_dpath, f)).pixel_array for f in os.listdir(r_dpath)]
        dcmimgs = np.array(_dcmimgs)
        print(r_dpath, r_lpath)
        lpath = os.path.join(r_lpath, os.listdir(r_lpath)[0])
        niftiimg = nibabel.load(lpath)
        dcmimgs = np.transpose(dcmimgs, [1, 2, 0])
        niftiimgs = np.transpose(niftiimg.get_fdata(), [1, 0, 2])
        print('Before :', dcmimgs.shape, niftiimgs.shape)
        dcm1_path = os.path.join(r_dpath, os.listdir(r_dpath)[0])
        dcm1 = pdcm.dcmread(dcm1_path)
        dcm_spacing = dcm1.PixelSpacing
        dcm_thickness = dcm1.SliceThickness
        dataset_space = [float(dcm_spacing[0]), float(dcm_spacing[1])]
        resampled_dcm = resample(dcmimgs, dataset_space)
        resampled_nii = resample(niftiimgs, dataset_space)
        print('After :', resampled_dcm.shape, resampled_nii.shape)
        new_d_path = r_dpath.replace('Dataset/Disc', 'Dataset/Disc_2D_resampled_nonthickness_dilate_closing')
        new_l_path = r_lpath.replace('Dataset/Disc', 'Dataset/Disc_2D_resampled_nonthickness_dilate_closing')
        for slice_n in range(resampled_dcm.shape[2]):
            slice_dcm = resampled_dcm[:, :, slice_n]
            slice_nii = resampled_nii[:, :, slice_n]
            kernel = np.ones((5, 5), np.uint8)
            slice_nii = cv2.morphologyEx(slice_nii, cv2.MORPH_DILATE, kernel)
            kernel = np.ones((3, 3), np.uint8)
            slice_nii = cv2.morphologyEx(slice_nii, cv2.MORPH_DILATE, kernel)
            kernel = np.ones((5, 5), np.uint8)
            slice_nii = cv2.morphologyEx(slice_nii, cv2.MORPH_CLOSE, kernel)

            file_name = str(0) * (4 - len(str(slice_n))) + str(slice_n) + '.png'
            slice_dcm_save_path = os.path.join(new_d_path, file_name)
            slice_nii_save_path = os.path.join(new_l_path, file_name)

            new_slice1_dcm = ((slice_dcm - np.min(slice_dcm)) / np.max(slice_dcm)) * 255
            pil_dcm_img = Image.fromarray(new_slice1_dcm.astype(np.uint8))
            pil_nii_img = Image.fromarray(slice_nii.astype(np.uint8) * 255)

            if not os.path.exists(os.path.dirname(slice_dcm_save_path)):
                os.makedirs(os.path.dirname(slice_dcm_save_path))
            if not os.path.exists(os.path.dirname(slice_nii_save_path)):
                os.makedirs(os.path.dirname(slice_nii_save_path))

            pil_dcm_img.save(slice_dcm_save_path)
            pil_nii_img.save(slice_nii_save_path)

    except:
        pass




## Not Resampling
# for r_dpath in [os.path.join('E:/Dataset/Disc/Dataset', d) for d in os.listdir('E:/Dataset/Disc/Dataset')]:
#     r_lpath = r_dpath.replace('Disc/Dataset', 'Disc/Label')
#     try:
#         _dcmimgs = [pdcm.dcmread(os.path.join(r_dpath, f)).pixel_array for f in os.listdir(r_dpath)]
#         dcmimgs = np.array(_dcmimgs)
#         print(r_dpath, r_lpath)
#         lpath = os.path.join(r_lpath, os.listdir(r_lpath)[0])
#         niftiimg = nibabel.load(lpath)
#         dcmimgs = np.transpose(dcmimgs, [1, 2, 0])
#         niftiimgs = np.transpose(niftiimg.get_fdata(), [1, 0, 2])
#         new_d_path = r_dpath.replace('Dataset/Disc', 'Dataset/Disc_2D')
#         new_l_path = r_lpath.replace('Dataset/Disc', 'Dataset/Disc_2D')
#         for slice_n in range(dcmimgs.shape[2]):
#             slice_dcm = dcmimgs[:, :, slice_n]
#             slice_nii = niftiimgs[:, :, slice_n]
#             file_name = str(0) * (4 - len(str(slice_n))) + str(slice_n) + '.png'
#             slice_dcm_save_path = os.path.join(new_d_path, file_name)
#             slice_nii_save_path = os.path.join(new_l_path, file_name)
#
#             new_slice1_dcm = ((slice_dcm - np.min(slice_dcm)) / np.max(slice_dcm)) * 255
#             pil_dcm_img = Image.fromarray(new_slice1_dcm.astype(np.uint8))
#             pil_nii_img = Image.fromarray(slice_nii.astype(np.uint8) * 255)
#
#             if not os.path.exists(os.path.dirname(slice_dcm_save_path)):
#                 os.makedirs(os.path.dirname(slice_dcm_save_path))
#             if not os.path.exists(os.path.dirname(slice_nii_save_path)):
#                 os.makedirs(os.path.dirname(slice_nii_save_path))
#
#             pil_dcm_img.save(slice_dcm_save_path)
#             pil_nii_img.save(slice_nii_save_path)
#     except:
#         pass

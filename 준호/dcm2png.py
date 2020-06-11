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
from pathlib import Path
import re


def try_int(input):
    """
    type changer if type(input) is not int
    """
    try:
        return int(input)
    except Exception as e:
        return input


def number_key(input):
    """
    make list that every value is int type number
    """
    return [try_int(val) for val in re.split('([0-9]+)', input)]


def sort_by_number(files):
    """
    sort list by using function number_key()
    """
    files.sort(key=number_key)
    return files


def instance_number_to_int(number):
    """
    get all numbers if input. if string, eliminate strings and return number only
    """
    try:
        return int(number)
    except:
        if type(number) == 'str':
            number = re.findall('([0-9]+)', number)
            return number
        else:
            raise TypeError


def list_sorter_by_inside_dict(list, reverse=False):
    new_list = []
    temp_list = []
    for row in list:
        fn = os.path.split(row)[-1]
        fn2 = os.path.dirname(row)
        # print(fn)
        # print(fn2)
        fp = os.path.join(fn2, fn)
        # print(fp)
        temp_list.append((fn, instance_number_to_int(pdcm.dcmread(fp).InstanceNumber)))

    _temp_list = sorted(temp_list, key=lambda x: x[1], reverse=reverse)
    for file_name, _ in _temp_list:
        for row in list:
            if file_name in row:
                new_list.append(row)
    return new_list


a = sort_by_number([os.path.join('E:/Dataset/Disc/Dataset', d) for d in os.listdir('E:/Dataset/Disc/Dataset')])
for r_dpath in a:
    b = list_sorter_by_inside_dict([os.path.join(r_dpath, f) for f in os.listdir(r_dpath)])
    try:
        r_lpath = r_dpath.replace('Disc/Dataset', 'Disc/Label')
        _dcmimgs = [pdcm.dcmread(os.path.join(r_dpath, f)).pixel_array for f in b]
        dcmimgs = np.array(_dcmimgs)
        print(r_dpath, r_lpath)
        lpath = os.path.join(r_lpath, os.listdir(r_lpath)[0])
        niftiimgs = nibabel.load(lpath)
        dcmimgs = np.transpose(dcmimgs, [1, 2, 0])
        niftiimgs = np.transpose(niftiimgs.get_fdata(), [1, 0, 2])
        # print(dcmimgs.shape, niftiimgs.shape)
        dcm1_path = os.path.join(r_dpath, os.listdir(r_dpath)[0])
        dcm1 = pdcm.dcmread(dcm1_path)
        if dcmimgs.shape != niftiimgs.shape:
            print(' ********* SHAPE WRONG', r_dpath, dcmimgs.shape, r_lpath, niftiimgs.shape)
        else:
            for z in range(dcmimgs.shape[-1]):
                dcmimg = dcmimgs[:,:,z]
                dcmimg = np.floor((dcmimg - np.min(dcmimg)) / np.max(dcmimg) * 255)

                niftiimg = niftiimgs[:,:,dcmimgs.shape[-1]-z-1]
                niftiimg = np.floor((niftiimg - np.min(niftiimg)) / np.max(niftiimg) * 255)

                dspath = os.path.join(r_dpath, str(0)*(4 - len(str(z+1)))+str(z+1)+'.png').replace('Disc', 'Disc_png')
                nspath = os.path.join(r_lpath, str(0)*(4 - len(str(z+1)))+str(z+1)+'.png').replace('Disc', 'Disc_png')
                # print('-', dspath, nspath)
                if not os.path.exists(os.path.dirname(dspath)):
                    os.makedirs(os.path.dirname(dspath))
                if not os.path.exists(os.path.dirname(nspath)):
                    os.makedirs(os.path.dirname(nspath))
                cv2.imwrite(dspath, dcmimg)
                cv2.imwrite(nspath, niftiimg)

    except Exception as ate:
        if 'TransferSyntaxUID' in str(ate):
            print(' ********* ', r_dpath, 'TransferSyntaxUID ERROR')
        else:
            print(' ********* ', r_dpath, ate)



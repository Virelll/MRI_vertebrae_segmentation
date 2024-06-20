# from main import *
import nibabel as nib
import numpy as np
import os


def load_data():
    images_384 = np.zeros((143, 384, 384), np.float32)  # 143
    masks_384 = np.zeros((143, 384, 384), np.float32)
    images_320 = np.zeros((85, 320, 320), np.float32)  # 85
    masks_320 = np.zeros((85, 320, 320), np.float32)
    k = 0
    k_320 = 0
    k_384 = 0
    for i in os.listdir('D:\\MRI\\NIFTI_files'):  # os.listdir('D:\\MRI\\NIFTI_files') ['C_spine_nifti']
        if "nifti" in str(i):
            img_path = 'D:\\MRI\\NIFTI_files\\' + str(i) + '\\images' + '\\'
            mask_path = 'D:\\MRI\\NIFTI_files\\' + str(i) + '\\masks' + '\\'
            for j in os.listdir(img_path):
                if 't1' in os.listdir(img_path + str(j))[0] or 't2' in os.listdir(img_path + str(j))[0]:
                    img = nib.load(img_path + str(j) + '\\' + os.listdir(img_path + str(j))[0]).get_fdata()
                    msk = nib.load(mask_path + str(j) + "\\" + os.listdir(mask_path + str(j))[0]).get_fdata()
                    for s in range(img.shape[2] // 2, img.shape[2] // 2 + 1, 1):
                        print("\n     ",img.shape)
                        if img.shape[0] == 384 and img.shape[1] == 384:
                            print("     ",img_path + str(j) + '\\' + os.listdir(img_path + str(j))[0])
                            images_384[k_384] = np.rot90(img[:, :, s] / img.max())
                            masks_384[k_384] = np.rot90(msk[:, :, s])
                            k_384 += 1
                            k += 1
                        elif img.shape[0] == 320 and img.shape[1] == 320:
                            print("     ",img_path + str(j) + '\\' + os.listdir(img_path + str(j))[0])
                            images_320[k_320] = np.rot90(img[:, :, s] / img.max())
                            masks_320[k_320] = np.rot90(msk[:, :, s])
                            k_320 += 1
                            k += 1
    print("     num images320 = ", k_320)
    print("     num images384 = ", k_384)
    print("     num images = ", k)

    return images_384, masks_384, images_320, masks_320

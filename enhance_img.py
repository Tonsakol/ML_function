import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means
import pywt

def noise_removal(IMG_PATH):
    img = cv2.imread(IMG_PATH)
    denoise = denoise_nl_means(img, patch_size=5, patch_distance=7, h=0.8,channel_axis=2)
    
    denoise = denoise * 255
    return denoise.astype(np.uint8)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def Norm(data):
    return ((data - np.min(data))*255) / (np.max(data) - np.min(data))

def enhance_img(IMG_PATH):
    denoise = noise_removal(IMG_PATH)

    (b, g, r) = cv2.split(denoise)
    # get v_in
    (b_cA, b_cD) = pywt.dwt(b, 'haar')
    (g_cA, g_cD) = pywt.dwt(g, 'haar')
    (r_cA, r_cD) = pywt.dwt(r, 'haar')
    b_ori = pywt.idwt(b_cA, b_cD, 'haar')
    g_ori = pywt.idwt(g_cA, g_cD, 'haar')
    r_ori = pywt.idwt(r_cA, r_cD, 'haar')

    img = cv2.merge((b_ori, g_ori, r_ori))
    img = img.astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_in = hsv_img[:, :, 2] 

    b_cA = Norm(b_cA)
    g_cA = Norm(g_cA)
    r_cA = Norm(r_cA)
    # apply clahe
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    b_cA = b_cA.astype(np.uint8)
    g_cA = g_cA.astype(np.uint8)
    r_cA = r_cA.astype(np.uint8)
    cl_b = clahe.apply(b_cA)
    cl_g = clahe.apply(g_cA)
    cl_r = clahe.apply(r_cA)

    b_re = pywt.idwt(cl_b, b_cD, 'haar')
    g_re = pywt.idwt(cl_g, g_cD, 'haar')
    r_re = pywt.idwt(cl_r, r_cD, 'haar')

    # get v_E
    re_img = cv2.merge((b_re, g_re, r_re))
    re_img = re_img.astype(np.uint8)
    hsv_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2HSV)
    v_E = hsv_img[:, :, 2]

    v_in = np.where(v_in==0,1,v_in)
    v_E = np.where(v_E==0,1,v_E)

    b_re = b_re*(v_E/ v_in)
    g_re = g_re*(v_E/ v_in)
    r_re = r_re*(v_E/ v_in)

    re_img = cv2.merge((b_re, g_re, r_re))
    b_norm = NormalizeData(b_re)
    g_norm = NormalizeData(g_re)
    r_norm = NormalizeData(r_re)

    M = np.ones(img.shape)
    norm_img = cv2.merge((b_norm, g_norm, r_norm))
    norm_img = norm_img**1.5
    en_img = (img * norm_img) + (1.5 * re_img * (M - norm_img))
    en_img = en_img.astype(np.uint8)
    #resize image
    desired_size = 299
    old_size = en_img.shape[:2] # old_size is in (height, width) format
    #print(old_size)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(en_img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


# IMG_PATH = 'D:\senior\capstone\ML_function\ISIC_0000013.jpg'
# img = cv2.imread(IMG_PATH)
# denoise = noise_removal(IMG_PATH)
# en_img = enhance_img(IMG_PATH)

# plt.imshow(en_img[:,:,::-1])
# plt.show()

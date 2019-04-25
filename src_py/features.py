import numpy as np
import cv2
from skimage import feature
import scipy
from matplotlib import pyplot as plt


def get_context_feature(img , crop_bbox):
    img_hist =cv2.calcHist([img], [0], None, [256], [0, 256])  # 图像直方图
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
    face = img[crop_bbox[0]:crop_bbox[2],crop_bbox[1]:crop_bbox[3]] # ？进行图像裁剪
    # cv2.rectangle(img, (crop_bbox[0], crop_bbox[1]),(crop_bbox[2], crop_bbox[3]), (0, 255, 0))
    # cv2.imshow('face',img)
    # cv2.waitKey(1)
    face_hist = cv2.calcHist([face], [0], None, [256], [0, 256])  # 人脸直方图
    nonface_hist = img_hist - face_hist
    face_hist = face_hist/sum(face_hist)           # 人脸直方图归一化
    nonface_hist = nonface_hist/sum(nonface_hist)  # 非人脸直方图归一化
    return face_hist.reshape(256),nonface_hist.reshape(256)


# def get_lighting_feature(img , crop_bbox):
#     face = img[crop_bbox[0]:crop_bbox[2],crop_bbox[1]:crop_bbox[3]]
#     face_hist = cv2.calcHist([face], [0], None, [256], [0, 256])  # 人脸直方图
#     face_hist = face_hist/sum(face_hist)           # 人脸直方图归一化
#     sum_x = 0
#     cnt = 0
#     index = []
#     for j in range(256):
#         if face_hist[j][0]!=0:
#             sum_x = sum_x + j
#             index.append(j)
#             cnt = cnt + 1
#     mu = sum_x / cnt
#     sigma = np.std(index)
#     x = np.linspace(1,256, 256)
#     # y = mlab.normpdf(x, mu, sigma)
#     y = scipy.stats.norm.pdf(x,mu,sigma)
#     return mu, sigma, y.reshape(256)

def get_lbp_feature(img,numPoints=8,radius=2):
    # img = cv2.equalizeHist(img)
    binsNumber = numPoints*(numPoints-1)+3 # 直方图bin的数量
    binsRange = np.arange(0,binsNumber+1) # 生成[0, binsNumber+1]的np.array，直方图x轴分隔点分布
    lbp = feature.local_binary_pattern(img, numPoints, radius, method = 'nri_uniform') #
    (hist, bins) = np.histogram(lbp.ravel(), bins=binsRange, range=(0,binsNumber),) # ？bins无用
    hist = hist/sum(hist) # 归一化
    return lbp, hist.reshape(binsNumber)
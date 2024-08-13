# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:45:48 2024

@author: jyshin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 Read
img = cv2.imread('C:/Users/jyshin/Desktop/vision/Model4.png', cv2.IMREAD_COLOR)
show_img = np.copy(img)

img_height, img_width = img.shape[:2]
y = int(img_height * 0.2)
x = int(img_width * 0.25)

w = int(img_width * 0.5)
h = int(img_height * 1.0)


labels = np.zeros(img.shape[:2], np.uint8)
# 그랩 컷 적용, 사각형을 기준으로 배경, 전경으로 분할
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, (x,y,w,h), None, None, 5, cv2.GC_INIT_WITH_RECT)

show_img = np.copy(img)
# 배경은 어둡게 표현
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] = 0

cv2.imshow('image',show_img)
cv2.waitKey()
cv2.destroyAllWindows()

image = show_img.astype(np.float32) / 255.
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB);  #컬러 공간 변환

data = image_lab.reshape((-1,3))

num_classes = 7 # k 값

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
# 픽셀을 num_classes 의 클러스터로 묶음
# 랜덤 센터 시작
_, labels, centers = cv2.kmeans(data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

segmented_lab = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2BGR)


plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('segmented')
plt.imshow(segmented)
plt.show()


cv2.imshow('segmented', segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()


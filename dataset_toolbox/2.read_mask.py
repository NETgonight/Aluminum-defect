import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像,BGR
mask_img = cv2.imread('aluminum/masks/23.png')

# 获取图像中不同的像素值
print(np.unique(mask_img))

# 只显示图像单通道
plt.imshow(mask_img[:, :, 0])
plt.axis('off')
plt.savefig('./aluminum/masklabel/23.png')
# 显示图像
plt.show()


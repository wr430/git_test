"""
opencv PIL numpy pytorch 处理图像数据
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

path = r"C:\Users\11578\Desktop\test\save_path_2D\2023_7_5-16_20_53-TOF1_depth_dst.jpg"

# opencv读取图片 (height, width, channels) 默认channels=3 BGR
cv_img = cv2.imread(path)
print(cv_img.shape)

# 转numpy数组
np_img = np.array(cv_img, dtype=float)
print(np_img.shape)

# 将opencv读取的图片转成tensor格式
# (channels, height, width)
tensor_image_1 = transforms.ToTensor()(cv_img)
print(tensor_image_1.size())
# (height, width, channels)
tensor_image_2 = torch.tensor(cv_img, dtype=torch.float)
print(tensor_image_2.size())

# 添加批维度  (batch, channels, height, width)
tensor_image = tensor_image_1.unsqueeze(0)
print(tensor_image.size())

# tensor
torch_tensor = torch.rand(1, 3, 240, 320)
print(torch_tensor.size())

# PIL读取图片
PIL_img = Image.open(path)
print(PIL_img.size)
# (width, height)
print(PIL_img.getbands())
# channels







import json
import os

import cv2
import numpy as np
from tqdm import tqdm

Dataset_Path = 'aluminum'

# 每个类别的信息及画mask的顺序（按照由大到小，由粗到精的顺序）
# 0-背景，从 1 开始
class_info = [
    {'label': 'zhe_zhou', 'type': 'polygon', 'color': 1},
    {'label': 'ca_shang', 'type': 'linestrip', 'color': 2, 'thickness': 10},
    {'label': 'zang_wu', 'type': 'polygon', 'color': 3},
    {'label': 'zhen_kong', 'type': 'polygon', 'color': 4},
]


def labelme2mask_single_img(img_path, labelme_json_path):
    """ 输入原始图像路径和labelme标注路径，输出 mask """
    img_bgr = cv2.imread(os.path.join(Dataset_Path, 'images', img_path))
    img_mask = np.zeros(img_bgr.shape[:2])  # 创建空白图像 0为背景

    img_json = os.path.join(Dataset_Path, 'json', labelme_json_path)
    with open(img_json, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    # 按顺序遍历每一个类别
    for one_class in class_info:
        # 遍历所有标注，找到属于当前类别的标注
        for each in labelme['shapes']:
            # 匹配类别
            if each['label'] == one_class['label']:
                # 获取点的坐标
                points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                # 多边形
                if one_class['type'] == 'polygon':
                    # 在空白图上画mask（闭合区域）
                    img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])
                # 线段
                elif one_class['type'] == 'line' or one_class['type'] == 'linestrip':
                    # 在空白图上画 mask（非闭合区域）
                    img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'], thickness=one_class['thickness'])
                else:
                    print('未知标注类型', one_class['type'])
    return img_mask


os.makedirs(os.path.join(Dataset_Path, 'masks'), exist_ok=True)
images_path = os.path.join(Dataset_Path, 'images')

for img_path in tqdm(os.listdir(images_path)):
    try:
        labelme_json_path = '.'.join(img_path.split('.')[:-1]) + '.json'

        img_mask = labelme2mask_single_img(img_path, labelme_json_path)

        mask_path = img_path.split('.')[0] + '.png'

        cv2.imwrite(os.path.join(Dataset_Path, 'masks', mask_path), img_mask)

    except Exception as E:
        print(img_path, '转换失败', E)


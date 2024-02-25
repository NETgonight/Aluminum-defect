import os
import shutil
import random

from tqdm import tqdm

Dataset_Path = 'aluminum'

os.makedirs(os.path.join('data', 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join('data', 'images', 'val'), exist_ok=True)
test_frac = 0.3   # 测试集比例
random.seed(666)  # 随机数种子，便于复现

image_dir = os.path.join(Dataset_Path, 'images')

img_paths = os.listdir(image_dir)
random.shuffle(img_paths)  # 随机打乱

val_number = int(len(img_paths) * test_frac)  # 测试集文件个数
train_files = img_paths[val_number:]          # 训练集文件名列表
val_files = img_paths[:val_number]            # 测试集文件名列表

print('数据集文件总数', len(img_paths))
print('训练集文件个数', len(train_files))
print('测试集文件个数', len(val_files))

for each in tqdm(train_files):
    src_path = os.path.join(image_dir, each)
    dst_path = os.path.join('data', 'images', 'train', each)
    shutil.copy(src_path, dst_path)


for each in tqdm(val_files):
    src_path = os.path.join(image_dir, each)
    dst_path = os.path.join('data', 'images', 'val', each)
    shutil.copy(src_path, dst_path)


# ----------------------------------

masks = 'masks'

os.makedirs(os.path.join('data', masks, 'train'), exist_ok=True)
os.makedirs(os.path.join('data', masks, 'val'), exist_ok=True)

for each in tqdm(train_files):
    src_path = os.path.join(Dataset_Path, masks, each.split('.')[0] + '.png')
    dst_path = os.path.join('data', masks, 'train', each.split('.')[0]+'.png')
    shutil.copy(src_path, dst_path)

for each in tqdm(val_files):
    src_path = os.path.join(Dataset_Path, masks, each.split('.')[0] + '.png')
    dst_path = os.path.join('data', masks, 'val', each.split('.')[0]+'.png')
    shutil.copy(src_path, dst_path)

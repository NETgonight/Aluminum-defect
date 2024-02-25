import os
import shutil
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.optim as opt
from torch.utils.data import DataLoader
from tools.my_dataset import CustomDataset
from tools.common_tools import get_net, CustomNetTrainer, get_logger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import config as cfg
import numpy as np

# torch.manual_seed(0)  # 对于PyTorch的随机数生成器
# np.random.seed(0)     # 对于NumPy的随机数生成器


BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 基础路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备


if __name__ == "__main__":
    # 网络配置
    data_dir = cfg.DATA_DIR      # 基础路径
    max_epoch = cfg.MAX_EPOCH    # 跑多少轮
    batch_size = cfg.BATCH_SIZE  # 每次载入多少图片
    model_path = cfg.MODEL_PATH  # 预训练模型

    # 优化器配置
    lr = cfg.LR                  # 学习率
    milestones = cfg.MILESTONES  # 学习率在第多少个epoch下降
    gamma = cfg.GAMMA            # 下降参数

    # 输出结果目录
    output_dir = cfg.LOG_DIR     # 结果保存路径
    log_name = cfg.log_name      # 日志文件路径

    # 标准化参数
    norm_mean = cfg.DATA_MEAN   # 均值
    norm_std = cfg.DATA_STD     # 方差

    # 若文件夹不存在,则创建
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir)  # 创建tensorboard文件
    shutil.copy("config.py", output_dir)    # 将当前配置文件拷贝一份到输出文件夹
    logger = get_logger(output_dir, log_name)   # 创建日志文件

    logger.info(f'Start | Model starts training!!!\n')

    # ============================ step 1/5 数据 ============================

    train_transform = transforms.Compose([          # 定义图片的预训练方式
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    label_transform = transforms.Compose([          # 定义标签的预训练方式
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
    ])

    # 构建DAGMDataset
    train_data = CustomDataset(data_dir=data_dir, mode='train', img_transform=train_transform, label_transform=label_transform)
    valid_data = CustomDataset(data_dir=data_dir, mode='val', img_transform=valid_transform, label_transform=label_transform)

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=cfg.DATALOADER_WORKERS)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, num_workers=cfg.DATALOADER_WORKERS)

    # ============================ step 2/5 模型 ============================

    custom_model = get_net(device=device, vis_model=False, path_state_dict=model_path)  # 获取预训练模型,不打印模型结构

    # ============================ step 3/5 损失函数 ============================

    # criterion = nn.CrossEntropyLoss()
    class_weights = torch.from_numpy(np.array([0.1, 0.2, 0.2, 0.2, 0.3])).float()
    class_weights = class_weights.to(device)  # device 是您的模型所在的设备
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ============================ step 4/5 优化器 ============================

    optimizer = opt.Adam(custom_model.parameters(), lr=lr)                                           # 优化器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=gamma, milestones=milestones)  # 设置学习率下降策略

    # ============================ step 5/5 训练 ============================

    start_time = datetime.now()        # 训练开始时间
    best_mIoU = 0
    for epoch in range(0, max_epoch):  # 模型开始训练

        # 得到train loss, valid loss
        loss_train = CustomNetTrainer.train(train_loader, custom_model, criterion, optimizer, epoch, device, max_epoch, logger)
        loss_valid, new_mIoU = CustomNetTrainer.valid(valid_loader, custom_model, criterion, epoch, device, max_epoch, logger)

        logger.info(f'Stage | Epoch[{epoch + 1:0>3}/{max_epoch:0>3}] Train loss:{loss_train:.8f}')
        logger.info(f'Stage | Epoch[{epoch + 1:0>3}/{max_epoch:0>3}] Valid loss:{loss_valid:.8f}')
        logger.info(f'Stage | Epoch[{epoch + 1:0>3}/{max_epoch:0>3}] LR:{optimizer.param_groups[0]["lr"]}\n')

        scheduler.step()  # 更新学习率

        # tensorboard绘制loss曲线和LR曲线
        writer.add_scalars("Loss", {"train": loss_train, "valid": loss_valid}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("MIoU", new_mIoU, epoch)

        if cfg.SAVE_MODEL:
            # 判断并保存模型
            if epoch == max_epoch - 1 or new_mIoU > best_mIoU:  # 追求完美,并保存最后一轮的模型
                checkpoint = {"CustomNet": custom_model.state_dict()}            # 模型参数
                if new_mIoU > best_mIoU:
                    # TODO writer.add_graph
                    best_mIoU = new_mIoU                                        # 更新best_mIoU
                path_checkpoint = os.path.join(output_dir, f"{cfg.TAG}_{epoch}e_{new_mIoU:.4%}.pkl")     # 模型保存路径
                torch.save(checkpoint, path_checkpoint)                         # 保存模型

    end_time = datetime.now()                     # 训练结束时间
    spend_time = (end_time - start_time).seconds  # 训练花费时间(s)

    logger.info(f'Final | Model training completed!!!')
    logger.info(f'Final | Start time: {datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Final | End time: {datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Final | Spend time: {spend_time}s')
    logger.info(f'Final | best_mIoU: {best_mIoU:.4%}')
    logger.info(f'Final | Final epoch is {max_epoch}')
    logger.info(f'Final | Each epoch spend {spend_time/max_epoch}s')

    writer.close()  # 关闭writer

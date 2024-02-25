import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径
DATA_DIR = os.path.join(BASE_DIR, "data")  # 数据集路径
DATA_MEAN = [0.16328026, 0.16328026, 0.16328026]    # 均值
DATA_STD = [0.2432042, 0.2432042, 0.2432042]     # 标准差

# MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")  # 预训练模型路径
MODEL_PATH = None
SAVE_MODEL = True


MAX_EPOCH = 10           # 跑多少轮
BATCH_SIZE = 4          # 每次载入多少图片
DATALOADER_WORKERS = 3  # dataloader线程数

TIME_STR = datetime.strftime(datetime.now(), '%m-%d-%H-%M')  # 时间格式化

LR = 0.01               # 学习率
MILESTONES = [100, 300, 700]        # 学习率在第多少个epoch下降
GAMMA = 0.1             # 下降参数

TAG = "Base_Light"        # 备注
LOG_DIR = os.path.join(BASE_DIR, "results", f"{TAG}_P{MAX_EPOCH}_B{BATCH_SIZE}_{TIME_STR}")  # 结果保存路径
log_name = f'{TIME_STR}.log'

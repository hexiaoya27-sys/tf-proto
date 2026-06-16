import torch
class Config:
    """配置文件类，统一管理所有参数"""

    def __init__(self):
        # 数据路径
        self.JSON_PATH = r'C:\Users\86157\PycharmProjects\metabg\data\sorted_sentences.json'
        self.CSV_PATH = r'C:\Users\86157\PycharmProjects\metabg\data\alldy.csv'

        # 模型路径
        self.MODEL_PATH = r'C:\Users\86157\PycharmProjects\hexiaoya\model\bert-base-chinese'

        # 数据处理参数
        self.NUM_SAMPLES = 4745
        self.MAX_LEN = 128
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 31

        # 训练参数
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.N_LABELS = 30
        self.INNER_LR = 0.01
        self.OUTER_LR = 1e-4
        self.STANDARD_LR = 1e-4

        # GAN参数
        # self.GAN_EPOCHS = 200
        # self.GAN_BATCH_SIZE = 128

        # 任务参数
        self.N_WAY = 5
        self.K_SHOT = 1
        self.QUERY_SIZE = 5
        self.NUM_TASKS = 40

        # 设备设置
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 输出路径
        self.MODEL_SAVE_PATH = 'best_model.pt'

    def display(self):
        """打印配置信息"""
        print("\n=== 配置参数 ===")
        for key, value in vars(self).items():
            print(f"{key}: {value}")
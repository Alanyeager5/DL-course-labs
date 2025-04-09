from dataclasses import dataclass


@dataclass
class BaseConfig:
    # 数据参数
    data_dir: str = "data/fb15k237"

    # 训练参数
    lr: float = 0.001
    batch_size: int = 1024
    eval_batch_size: int = 2048
    max_epochs: int = 200
    patience: int = 10
    margin: float = 1.0

    # 模型参数
    embedding_dim: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 评估参数
    eval_every: int = 1  # 每N个epoch评估一次
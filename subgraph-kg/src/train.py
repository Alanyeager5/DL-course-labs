import torch
import numpy as np
from torch.optim import AdamW, lr_scheduler
from .evaluate import Evaluator


class KGTrainer:
    """知识图谱训练引擎"""

    def __init__(self, model, train_data, valid_data, config):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.config = config
        self.evaluator = Evaluator()

        # 优化器配置
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', patience=3
        )

        # 训练状态
        self.best_metric = 0
        self.early_stop_counter = 0

    def train_epoch(self):
        """单个训练epoch"""
        self.model.train()
        total_loss = 0

        for h, r, t in DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True
        ):
            # 负采样和损失计算
            neg_t = self._generate_negative_samples(h.size(0))
            loss = self._compute_loss(h, r, t, neg_t)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_data)

    def _generate_negative_samples(self, batch_size):
        """随机负采样（可重写实现高级策略）"""
        return torch.randint(
            0, len(self.model.entity_emb.weight),
            (batch_size,),
            device=self.config.device
        )

    def _compute_loss(self, h, r, t, neg_t):
        """边际排序损失计算"""
        pos_scores = self.model(h, r, t)
        neg_scores = self.model(h, r, neg_t)
        return torch.relu(self.config.margin + neg_scores - pos_scores).mean()

    def train(self):
        """完整训练流程"""
        for epoch in range(self.config.max_epochs):
            # 训练阶段
            train_loss = self.train_epoch()

            # 验证评估
            valid_metrics = self.evaluator.evaluate(
                self.model, self.valid_data, self.config
            )

            # 学习率调整
            self.scheduler.step(valid_metrics['MRR'])

            # 打印日志
            self._log_progress(epoch, train_loss, valid_metrics)

            # 早停判断
            if self._check_early_stop(valid_metrics):
                break

    def _log_progress(self, epoch, loss, metrics):
        """格式化输出训练信息"""
        print(f"Epoch {epoch + 1}/{self.config.max_epochs}")
        print(f"Train Loss: {loss:.4f} | Valid MRR: {metrics['MRR']:.4f} "
              f"| Hit@10: {metrics['Hit@10']:.4f} | AUC: {metrics['AUC']:.4f}")

    def _check_early_stop(self, metrics):
        """早停判断"""
        if metrics['MRR'] > self.best_metric:
            self.best_metric = metrics['MRR']
            torch.save(self.model.state_dict(), 'best_model.pt')
            self.early_stop_counter = 0
            return False
        self.early_stop_counter += 1
        return self.early_stop_counter >= self.config.patience
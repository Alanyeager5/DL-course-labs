import torch.nn as nn
import torch.nn.functional as F


class BaseKGModel(nn.Module):
    """知识图谱模型基类"""
    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score_function(self, h, r, t):
        """需子类实现的具体评分函数"""
        raise NotImplementedError

    def forward(self, h, r, t=None):

        h_emb = self.entity_emb(h.squeeze())
        r_emb = self.relation_emb(r.squeeze())

        if t is None:
            return self.score_function(h_emb, r_emb, self.entity_emb.weight)
        t_emb = self.entity_emb(t.squeeze())
        return self.score_function(h_emb, r_emb, t_emb)


class TransE(BaseKGModel):
    """TransE模型实现"""

    def __init__(self, num_entities, num_relations, embed_dim, norm=1):
        super().__init__(num_entities, num_relations, embed_dim)
        self.norm = norm

    def score_function(self, h, r, t):
        return -torch.norm(h + r - t, p=self.norm, dim=-1)


class GNNModel(BaseKGModel):
    """GNN模型基类（需实现具体图网络）"""

    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__(num_entities, num_relations, embed_dim)
        # 在此添加GNN层定义
import os
import torch
from torch.utils.data import Dataset, DataLoader

class KGDataset(Dataset):
    """三元组数据集处理"""
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return (
            torch.LongTensor([self.entity2id[h]]),
            torch.LongTensor([self.relation2id[r]]),
            torch.LongTensor([self.entity2id[t]])
        )


class DataProcessor:
    """数据预处理"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id, self.relation2id = self._build_vocab()

    def _load_triples(self, filename):
        with open(os.path.join(self.data_dir, filename)) as f:
            return [line.strip().split('\t') for line in f]

    def _build_vocab(self):
        entities, relations = set(), set()
        for split in ['train', 'valid', 'test']:
            for h, r, t in self._load_triples(f"{split}.txt"):
                entities.update({h, t})
                relations.add(r)
        return (
            {e: i for i, e in enumerate(entities)},
            {r: i for i, r in enumerate(relations)}
        )

    def get_datasets(self):
        """获取三元组数据集"""
        return {
            split: KGDataset(
                self._load_triples(f"{split}.txt"),
                self.entity2id,
                self.relation2id
            ) for split in ['train', 'valid', 'test']
        }
import numpy as np
from tqdm import tqdm


class Evaluator:

    def evaluate(self, model, dataset, config):
        model.eval()
        ranks, pos_scores, neg_scores = [], [], []

        with torch.no_grad():
            for h, r, t in tqdm(DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size,
                    pin_memory=True
            )):
                # 正样本得分
                pos_pred = model(h, r, t).cpu().numpy()
                pos_scores.append(pos_pred)

                # 负样本得分
                neg_t = self._generate_negative_samples(h.size(0), config)
                neg_pred = model(h, r, neg_t).cpu().numpy()
                neg_scores.append(neg_pred)

                # 全量排名计算
                all_scores = model(h, r).cpu().numpy()
                ranks.extend(self._calculate_ranks(all_scores, t.cpu().numpy()))

        # 指标计算
        return {
            'MRR': self._calculate_mrr(ranks),
            'Hit@10': self._calculate_hitk(ranks, 10),
            'AUC': self._calculate_auc(
                np.concatenate(pos_scores),
                np.concatenate(neg_scores)
            )
        }

    def _calculate_ranks(self, scores, targets):
        """批量计算排名"""
        return [np.where(np.argsort(-row) == target)[0][0] + 1
                for row, target in zip(scores, targets)]

    def _calculate_mrr(self, ranks):
        return np.mean(1.0 / np.array(ranks))

    def _calculate_hitk(self, ranks, k):
        return np.mean(np.array(ranks) <= k)

    def _calculate_auc(self, pos, neg):
        """AUC计算"""
        pos = pos[:, None]
        return (pos > neg[None, :]).mean() + 0.5 * (pos == neg[None, :]).mean()
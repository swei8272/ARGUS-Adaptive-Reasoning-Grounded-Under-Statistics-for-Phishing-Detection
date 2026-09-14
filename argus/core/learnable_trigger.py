"""
可学习的任务触发器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class LearnableTrigger(nn.Module):
    """
    可学习的任务触发器
    结合先验规则和神经网络学习的残差
    """

    def __init__(self,
                 feature_dim: int = 100,
                 num_tasks: int = 18,
                 hidden_dim: int = 64,
                 prior_weight_init: float = 0.7):
        """
        Args:
            feature_dim: 特征维度
            num_tasks: 任务数量
            hidden_dim: 隐藏层维度
            prior_weight_init: 先验权重初始值
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_tasks = num_tasks

        # 残差网络（学习先验规则的修正）
        self.residual_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_tasks),
            nn.Tanh()  # 输出 [-1, 1] 的残差
        )

        # 融合权重（可学习）
        # 使用logit形式，通过sigmoid映射到[0,1]
        self.alpha_logit = nn.Parameter(
            torch.tensor(self._inverse_sigmoid(prior_weight_init))
        )

        # 残差缩放系数（限制残差影响）
        self.residual_scale = 0.3

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """sigmoid的反函数"""
        x = max(min(x, 0.9999), 0.0001)  # 避免数值问题
        return torch.log(torch.tensor(x / (1 - x))).item()

    def forward(self, features: torch.Tensor, prior_scores: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: (batch_size, feature_dim) 特征向量
            prior_scores: (batch_size, num_tasks) 先验触发得分

        Returns:
            final_scores: (batch_size, num_tasks) 最终触发得分 [0, 1]
        """
        # 计算可学习残差
        residual = self.residual_net(features)  # (batch, num_tasks), [-1, 1]

        # 计算融合权重
        alpha = torch.sigmoid(self.alpha_logit)  # [0, 1]

        # 融合：先验 + 缩放的残差
        # final = α * prior + (1-α) * (prior + scale * residual)
        final_scores = alpha * prior_scores + \
                       (1 - alpha) * (prior_scores + self.residual_scale * residual)

        # Clamp到[0, 1]
        final_scores = torch.clamp(final_scores, 0, 1)

        return final_scores

    def get_top_k_tasks(self,
                        features: torch.Tensor,
                        prior_scores: torch.Tensor,
                        k: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取top-k任务

        Args:
            features: (batch_size, feature_dim)
            prior_scores: (batch_size, num_tasks)
            k: 选择的任务数量

        Returns:
            top_k_scores: (batch_size, k) Top-k得分
            top_k_indices: (batch_size, k) Top-k任务索引
        """
        final_scores = self.forward(features, prior_scores)
        top_k_scores, top_k_indices = torch.topk(final_scores, k, dim=1)
        return top_k_scores, top_k_indices

    def get_interpretability_info(self) -> Dict:
        """
        获取可解释性信息（用于论文展示）

        Returns:
            info: Dict包含先验权重、可学习权重等
        """
        alpha = torch.sigmoid(self.alpha_logit).item()
        return {
            'prior_weight': alpha,
            'learned_weight': 1 - alpha,
            'residual_scale': self.residual_scale,
            'alpha_logit': self.alpha_logit.item()
        }


class TaskRelevanceTracker:
    """
    任务相关性跟踪器
    用于计算每个任务对最终预测的贡献度（训练监督信号）
    """

    def __init__(self):
        self.contributions = []

    def compute_task_contribution(self,
                                  model: nn.Module,
                                  features: torch.Tensor,
                                  triggered_task_indices: torch.Tensor,
                                  final_prediction: torch.Tensor,
                                  ground_truth: torch.Tensor) -> torch.Tensor:
        """
        计算任务贡献度（使用leave-one-out方法）

        Args:
            model: 完整检测模型
            features: (batch_size, feature_dim)
            triggered_task_indices: (batch_size, k) 触发的任务索引
            final_prediction: (batch_size,) 最终预测概率
            ground_truth: (batch_size,) 真实标签

        Returns:
            contributions: (batch_size, num_tasks) 每个任务的贡献度
        """
        batch_size = features.shape[0]
        num_tasks = model.num_tasks if hasattr(model, 'num_tasks') else 18
        contributions = torch.zeros(batch_size, num_tasks, device=features.device)

        # 对每个触发的任务
        for task_idx in range(num_tasks):
            # 创建mask（排除当前任务）
            task_mask = triggered_task_indices != task_idx

            # 使用剩余任务重新预测
            with torch.no_grad():
                masked_pred = self._predict_with_mask(model, features, task_mask)

            # 计算预测变化
            # 如果移除该任务后，预测变差 → 该任务重要
            delta_accuracy = self._compute_accuracy_delta(
                final_prediction, masked_pred, ground_truth
            )

            # 只记录实际触发的任务的贡献
            is_triggered = (triggered_task_indices == task_idx).any(dim=1)
            contributions[:, task_idx] = delta_accuracy * is_triggered.float()

        return contributions

    @staticmethod
    def _predict_with_mask(model, features, task_mask):
        """使用部分任务进行预测（简化版本）"""
        # 这里需要根据你的模型架构具体实现
        # 假设模型有一个partial_forward方法
        if hasattr(model, 'partial_forward'):
            return model.partial_forward(features, task_mask)
        else:
            # Fallback: 返回原始预测
            return model(features)

    @staticmethod
    def _compute_accuracy_delta(pred1, pred2, ground_truth):
        """计算准确率差异"""
        acc1 = ((pred1 > 0.5) == ground_truth).float()
        acc2 = ((pred2 > 0.5) == ground_truth).float()
        return acc1 - acc2  # 正值表示移除任务后准确率下降
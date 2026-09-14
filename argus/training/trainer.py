"""
ARGUS训练器 - 联合优化触发器、置信度估计器和融合器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from datetime import datetime


class ARGUSTrainer:
    """
    ARGUS完整训练器

    联合优化：
    1. LearnableTrigger: 任务触发器
    2. ConfidenceEstimator: 置信度估计
    3. MultimodalFusion: 多模态融合
    """

    def __init__(self,
                 learnable_trigger,
                 confidence_estimator,
                 multimodal_fusion,
                 config):
        """
        Args:
            learnable_trigger: 可学习触发器
            confidence_estimator: 置信度估计器
            multimodal_fusion: 多模态融合器
            config: 配置对象
        """
        self.trigger = learnable_trigger
        self.confidence = confidence_estimator
        self.fusion = multimodal_fusion
        self.config = config

        # 优化器
        self.optimizer = optim.AdamW([
            {'params': self.trigger.parameters(), 'lr': config.learning_rate},
            {'params': self.confidence.parameters(), 'lr': config.learning_rate},
            {'params': self.fusion.parameters(), 'lr': config.learning_rate}
        ], weight_decay=config.weight_decay)

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )

        # 损失函数
        self.detection_criterion = nn.BCEWithLogitsLoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

    def compute_loss(self,
                     features: torch.Tensor,
                     prior_scores: torch.Tensor,
                     labels: torch.Tensor,
                     risk_scores: Optional[Dict[str, torch.Tensor]] = None,
                     confidences: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Args:
            features: (batch_size, num_features) 特征
            prior_scores: (batch_size, num_tasks) 先验得分
            labels: (batch_size,) 标签 [0, 1]
            risk_scores: 各模态风险评分（可选）
            confidences: 各模态置信度（可选）

        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        batch_size = features.shape[0]

        # 1. 触发任务
        trigger_scores, task_indices = self.trigger.get_top_k_tasks(
            features, prior_scores, k=self.config.top_k_tasks
        )

        # 2. 计算置信度（仅用quantitative）
        conf_dict = self.confidence.compute_quantitative_confidence(features)
        quant_confidence = conf_dict['confidence']

        # 3. 如果没有提供risk_scores，用简单规则计算
        if risk_scores is None:
            # 使用规则特征的均值作为quantitative risk
            rule_features = features[:, 82:92]  # 规则特征索引82-91
            quant_risk = torch.sigmoid(rule_features.mean(dim=1))

            risk_scores = {
                'quantitative': quant_risk,
                'visual': torch.ones(batch_size, device=features.device) * 0.5,  # 占位
                'semantic': torch.ones(batch_size, device=features.device) * 0.5  # 占位
            }

        # 4. 如果没有提供confidences
        if confidences is None:
            confidences = {
                'quantitative': quant_confidence,
                'visual': torch.ones(batch_size, device=features.device) * 0.8,
                'semantic': torch.ones(batch_size, device=features.device) * 0.8
            }

        # 5. 融合
        fused_score = self.fusion(risk_scores, confidences)

        # 6. 检测损失（主要损失）
        detection_loss = self.detection_criterion(
            fused_score,
            labels.float()
        )

        # 7. 效率惩罚（惩罚触发过多任务）
        # 期望：平均触发k个任务（默认6个）
        efficiency_penalty = torch.abs(
            trigger_scores.sum(dim=1).mean() - self.config.top_k_tasks
        )

        # 8. 相关性奖励（奖励高分任务）
        # 如果预测正确，奖励高触发分数；如果预测错误，惩罚
        predictions = (torch.sigmoid(fused_score) > 0.5).float()
        correct = (predictions == labels.float()).float()

        # 正确预测时，奖励高分任务；错误预测时，不奖励
        relevance_reward = (trigger_scores.max(dim=1)[0] * correct).mean()

        # 9. 总损失
        total_loss = (
                self.config.detection_loss_weight * detection_loss +
                self.config.efficiency_penalty_weight * efficiency_penalty -
                self.config.relevance_reward_weight * relevance_reward
        )

        # 损失详情
        loss_dict = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'efficiency_penalty': efficiency_penalty.item(),
            'relevance_reward': relevance_reward.item(),
            'avg_tasks_triggered': trigger_scores.sum(dim=1).mean().item()
        }

        return total_loss, loss_dict

    def train_epoch(self, train_loader, epoch: int) -> Dict:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch

        Returns:
            metrics: 训练指标
        """
        self.trigger.train()
        self.confidence.train()
        self.fusion.train()

        total_loss = 0.0
        total_detection_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            # 提取数据
            features = batch['features'].to(self.config.device)
            prior_scores = batch['prior_scores'].to(self.config.device)
            labels = batch['label'].to(self.config.device)

            # 前向传播
            loss, loss_dict = self.compute_loss(
                features, prior_scores, labels
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.trigger.parameters()) +
                list(self.confidence.parameters()) +
                list(self.fusion.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            # 统计
            total_loss += loss_dict['total_loss']
            total_detection_loss += loss_dict['detection_loss']
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'det_loss': f"{loss_dict['detection_loss']:.4f}",
                'tasks': f"{loss_dict['avg_tasks_triggered']:.1f}"
            })

        # 更新学习率
        self.scheduler.step()

        metrics = {
            'avg_loss': total_loss / num_batches,
            'avg_detection_loss': total_detection_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        self.history['train_loss'].append(metrics['avg_loss'])

        return metrics

    def evaluate(self, val_loader) -> Dict:
        """
        评估模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            metrics: 评估指标
        """
        self.trigger.eval()
        self.confidence.eval()
        self.fusion.eval()

        all_predictions = []
        all_labels = []
        all_scores = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                features = batch['features'].to(self.config.device)
                prior_scores = batch['prior_scores'].to(self.config.device)
                labels = batch['label'].to(self.config.device)

                # 前向传播
                loss, loss_dict = self.compute_loss(
                    features, prior_scores, labels
                )

                # 获取预测
                # 重新计算（因为compute_loss内部已经计算过了）
                trigger_scores, _ = self.trigger.get_top_k_tasks(
                    features, prior_scores, k=self.config.top_k_tasks
                )

                conf_dict = self.confidence.compute_quantitative_confidence(features)

                # 简单风险评分
                rule_features = features[:, 82:92]
                quant_risk = torch.sigmoid(rule_features.mean(dim=1))

                risk_scores = {
                    'quantitative': quant_risk,
                    'visual': torch.ones_like(quant_risk) * 0.5,
                    'semantic': torch.ones_like(quant_risk) * 0.5
                }

                confidences = {
                    'quantitative': conf_dict['confidence'],
                    'visual': torch.ones_like(quant_risk) * 0.8,
                    'semantic': torch.ones_like(quant_risk) * 0.8
                }

                fused_score = self.fusion(risk_scores, confidences)
                predictions = (torch.sigmoid(fused_score) > 0.5).float()

                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(torch.sigmoid(fused_score).cpu().numpy())

                total_loss += loss_dict['total_loss']
                num_batches += 1

        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = (all_predictions == all_labels).mean()

        # Precision, Recall, F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'avg_loss': total_loss / num_batches,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        self.history['val_loss'].append(metrics['avg_loss'])
        self.history['val_accuracy'].append(accuracy)
        self.history['val_f1'].append(f1)

        return metrics

    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        """
        保存检查点

        Args:
            filepath: 保存路径
            epoch: 当前epoch
            metrics: 当前指标
        """
        checkpoint = {
            'epoch': epoch,
            'trigger_state_dict': self.trigger.state_dict(),
            'confidence_state_dict': self.confidence.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': {
                'num_features': self.config.num_features,
                'num_tasks': self.config.num_tasks,
                'top_k_tasks': self.config.top_k_tasks
            }
        }

        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        加载检查点

        Args:
            filepath: 检查点路径
        """
        checkpoint = torch.load(filepath, map_location=self.config.device)

        self.trigger.load_state_dict(checkpoint['trigger_state_dict'])
        self.confidence.load_state_dict(checkpoint['confidence_state_dict'])
        self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)

        print(f"✅ Checkpoint loaded from {filepath}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint['epoch']
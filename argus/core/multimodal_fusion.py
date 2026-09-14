"""
Omni-Modal多模态融合器
支持任意模态缺失，自动权重调整
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import warnings


class OmniModalFusion(nn.Module):
    """
    全模态自适应融合器

    特性：
    - 支持任意模态组合（至少1个）
    - 自动重新归一化权重
    - 缺失模态的优雅降级
    """

    def __init__(self,
                 initial_weights: Optional[Dict[str, float]] = None,
                 modality_names: Optional[List[str]] = None,
                 min_confidence_threshold: float = 0.01):
        """
        Args:
            initial_weights: 初始模态权重（可选）
            modality_names: 支持的模态名称列表
            min_confidence_threshold: 最小置信度阈值
        """
        super().__init__()

        # 默认支持的模态
        if modality_names is None:
            modality_names = ['quantitative', 'visual', 'semantic']
        self.modality_names = modality_names

        # 默认权重（均匀分布）
        if initial_weights is None:
            initial_weights = {name: 1.0 / len(modality_names) for name in modality_names}

        # 可学习的基础权重（logits形式，通过softmax归一化）
        self.weight_logits = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(self._inv_softmax(initial_weights.get(name, 1.0 / len(modality_names)))))
            for name in modality_names
        })

        # 最小置信度阈值
        self.min_confidence_threshold = min_confidence_threshold

    @staticmethod
    def _inv_softmax(p: float, num_classes: int = 3) -> float:
        """softmax的近似反函数"""
        eps = 1e-6
        p = max(min(p, 1.0 - eps), eps)
        return torch.log(torch.tensor(p * num_classes)).item()

    def get_base_weights(self) -> Dict[str, float]:
        """
        获取基础权重（未经置信度调整）

        Returns:
            weights: Dict[modality_name, weight]
        """
        logits = torch.stack([self.weight_logits[name] for name in self.modality_names])
        weights = torch.softmax(logits, dim=0)

        return {name: weights[i].item() for i, name in enumerate(self.modality_names)}

    def _extract_available_modalities(self,
                                     risk_scores: Dict[str, torch.Tensor],
                                     confidences: Dict[str, torch.Tensor]) -> Tuple[List[str], Dict]:
        """
        提取可用的模态

        Args:
            risk_scores: 风险评分字典
            confidences: 置信度字典

        Returns:
            available_modalities: 可用模态列表
            availability_info: 可用性信息
        """
        available = []
        availability_info = {
            'available': [],
            'missing': [],
            'low_confidence': []
        }

        for modality in self.modality_names:
            # 检查是否存在
            has_score = modality in risk_scores and risk_scores[modality] is not None
            has_confidence = modality in confidences and confidences[modality] is not None

            if not has_score and not has_confidence:
                availability_info['missing'].append(modality)
                continue

            # 检查置信度是否足够
            if has_confidence:
                conf = confidences[modality]
                if isinstance(conf, torch.Tensor):
                    avg_conf = conf.mean().item()
                else:
                    avg_conf = float(conf)

                if avg_conf < self.min_confidence_threshold:
                    availability_info['low_confidence'].append(modality)
                    continue

            # 可用
            available.append(modality)
            availability_info['available'].append(modality)

        return available, availability_info

    def _normalize_tensor(self, tensor, default_value: float = 0.5) -> torch.Tensor:
        """
        安全地归一化张量

        Args:
            tensor: 输入张量（可能是None/float/int/tensor）
            default_value: 如果无法归一化的默认值

        Returns:
            normalized_tensor: torch.Tensor
        """
        if tensor is None:
            return torch.tensor(default_value)

        if isinstance(tensor, (int, float)):
            return torch.tensor(float(tensor))

        if not isinstance(tensor, torch.Tensor):
            try:
                tensor = torch.tensor(tensor)
            except:
                return torch.tensor(default_value)

        return tensor

    def _ensure_same_shape(self, tensors: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
        """
        确保所有张量形状一致

        Args:
            tensors: 张量字典
            batch_size: 目标batch size

        Returns:
            normalized_tensors: 形状一致的张量字典
        """
        result = {}

        for key, tensor in tensors.items():
            if tensor.ndim == 0:
                # 标量 -> (batch_size,)
                result[key] = tensor.unsqueeze(0).expand(batch_size)
            elif tensor.ndim == 1 and tensor.shape[0] == 1:
                # (1,) -> (batch_size,)
                result[key] = tensor.expand(batch_size)
            elif tensor.ndim == 1 and tensor.shape[0] == batch_size:
                # (batch_size,) -> 保持
                result[key] = tensor
            else:
                # 其他情况：取平均后扩展
                result[key] = torch.full((batch_size,), tensor.mean().item())

        return result

    def fuse_with_confidence(self,
                            risk_scores: Dict[str, torch.Tensor],
                            confidences: Dict[str, torch.Tensor],
                            adjusted_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Omni-Modal置信度加权融合

        Args:
            risk_scores: Dict[modality, (batch_size,)] 各模态的风险评分
            confidences: Dict[modality, (batch_size,)] 各模态的置信度
            adjusted_weights: 冲突消解后调整的权重（可选）

        Returns:
            fused_score: (batch_size,) 融合后的风险评分
            fusion_info: Dict包含融合详细信息
        """
        # 1. 检测可用模态
        available_modalities, availability_info = self._extract_available_modalities(
            risk_scores, confidences
        )

        # 2. 至少需要一个模态
        if not available_modalities:
            warnings.warn("No available modalities! Returning default score 0.5")

            # 尝试推断batch_size
            batch_size = 1
            for scores in risk_scores.values():
                if scores is not None and isinstance(scores, torch.Tensor) and scores.numel() > 0:
                    batch_size = scores.shape[0] if scores.ndim > 0 else 1
                    break

            return torch.full((batch_size,), 0.5), {
                'error': 'no_available_modalities',
                'availability': availability_info,
                'fallback_score': 0.5
            }

        # 3. 获取基础权重
        if adjusted_weights is None:
            base_weights = self.get_base_weights()
        else:
            base_weights = adjusted_weights.copy()

        # 4. 提取并归一化可用模态的数据
        batch_size = None
        raw_scores = {}
        raw_confs = {}
        raw_weights = {}

        # 第一遍：提取所有数据并推断batch_size
        for modality in available_modalities:
            score = self._normalize_tensor(risk_scores.get(modality, 0.5))
            conf = self._normalize_tensor(confidences.get(modality, 1.0))
            weight = base_weights.get(modality, 1.0 / len(available_modalities))

            # 推断batch_size
            if batch_size is None:
                if score.ndim > 0 and score.numel() > 0:
                    batch_size = score.shape[0]
                elif conf.ndim > 0 and conf.numel() > 0:
                    batch_size = conf.shape[0]

            raw_scores[modality] = score
            raw_confs[modality] = conf
            raw_weights[modality] = weight

        # 如果还没推断出batch_size，设为1
        if batch_size is None:
            batch_size = 1

        # 第二遍：统一形状
        normalized_scores = self._ensure_same_shape(raw_scores, batch_size)
        normalized_confs = self._ensure_same_shape(raw_confs, batch_size)

        # 5. 置信度加权融合
        weighted_sum = torch.zeros(batch_size)
        weight_sum = torch.zeros(batch_size)

        for modality in available_modalities:
            score = normalized_scores[modality]
            conf = normalized_confs[modality]
            weight = raw_weights[modality]

            effective_weight = weight * conf
            weighted_sum += effective_weight * score
            weight_sum += effective_weight

        # 6. 归一化（避免除零）
        weight_sum = torch.clamp(weight_sum, min=1e-6)
        fused_score = weighted_sum / weight_sum

        # 7. 构建融合信息（用于解释）
        effective_weights = {}
        for modality in available_modalities:
            conf = normalized_confs[modality]
            weight = raw_weights[modality]
            eff_weight = (weight * conf / weight_sum).mean().item()
            effective_weights[modality] = eff_weight

        fusion_info = {
            'available_modalities': available_modalities,
            'missing_modalities': availability_info['missing'],
            'low_confidence_modalities': availability_info['low_confidence'],
            'base_weights': {m: base_weights.get(m, 0.0) for m in self.modality_names},
            'effective_weights': effective_weights,
            'risk_scores': {m: normalized_scores[m].mean().item() for m in available_modalities},
            'confidences': {m: normalized_confs[m].mean().item() for m in available_modalities},
            'weight_renormalization': len(available_modalities) < len(self.modality_names)
        }

        return fused_score, fusion_info

    def forward(self,
                risk_scores: Dict[str, torch.Tensor],
                confidences: Dict[str, torch.Tensor],
                risk_adjustment: float = 0.0,
                adjusted_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        前向传播（简化接口）

        Args:
            risk_scores: 各模态风险评分（可缺失）
            confidences: 各模态置信度（可缺失）
            risk_adjustment: 冲突消解的风险调整量
            adjusted_weights: 调整后的权重（可选）

        Returns:
            final_score: 最终风险评分 [0, 1]
        """
        fused_score, fusion_info = self.fuse_with_confidence(
            risk_scores, confidences, adjusted_weights
        )

        # 应用风险调整
        final_score = torch.clamp(fused_score + risk_adjustment, 0.0, 1.0)

        # 训练时警告（可选）
        if self.training and fusion_info.get('weight_renormalization', False):
            missing = fusion_info['missing_modalities']
            if missing and len(missing) > 0:
                warnings.warn(f"Missing modalities during training: {missing}", stacklevel=2)

        return final_score

    def explain_fusion(self, fusion_info: Dict) -> str:
        """
        生成融合过程的可读解释

        Args:
            fusion_info: fuse_with_confidence返回的融合信息

        Returns:
            explanation: 文字解释
        """
        lines = ["=== Fusion Explanation ==="]

        # 可用性状态
        available = fusion_info.get('available_modalities', [])
        missing = fusion_info.get('missing_modalities', [])
        low_conf = fusion_info.get('low_confidence_modalities', [])

        lines.append(f"\n✅ Available: {', '.join(available) if available else 'None'}")

        if missing:
            lines.append(f"❌ Missing: {', '.join(missing)}")

        if low_conf:
            lines.append(f"⚠️  Low confidence: {', '.join(low_conf)}")

        # 权重信息
        if available:
            lines.append("\n--- Weight Distribution ---")
            eff_weights = fusion_info.get('effective_weights', {})
            base_weights = fusion_info.get('base_weights', {})
            for modality in available:
                base_w = base_weights.get(modality, 0.0)
                eff_w = eff_weights.get(modality, 0.0)
                lines.append(f"{modality}: {base_w:.3f} (base) → {eff_w:.3f} (effective)")

            # 评分信息
            lines.append("\n--- Risk Scores ---")
            scores = fusion_info.get('risk_scores', {})
            confs = fusion_info.get('confidences', {})
            for modality in available:
                score = scores.get(modality, 0.0)
                conf = confs.get(modality, 0.0)
                lines.append(f"{modality}: score={score:.3f}, confidence={conf:.3f}")

        # 重归一化提示
        if fusion_info.get('weight_renormalization', False):
            lines.append("\n⚠️  Weights were renormalized due to missing modalities")

        return "\n".join(lines)

    def get_modality_importance(self,
                               risk_scores: Dict[str, torch.Tensor],
                               confidences: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算每个模态的重要性（用于可解释性）

        Args:
            risk_scores: 风险评分
            confidences: 置信度

        Returns:
            importance: Dict[modality, importance_score]
        """
        _, fusion_info = self.fuse_with_confidence(risk_scores, confidences)

        importance = {}
        eff_weights = fusion_info.get('effective_weights', {})
        scores = fusion_info.get('risk_scores', {})

        for modality in fusion_info.get('available_modalities', []):
            weight = eff_weights.get(modality, 0.0)
            score = scores.get(modality, 0.5)
            deviation = abs(score - 0.5)

            importance[modality] = weight * (1 + deviation)

        return importance


class OmniModalFusionWithFallback(OmniModalFusion):
    """
    带降级策略的Omni-Modal融合器

    额外特性：
    - 定义模态的fallback顺序
    - 当高优先级模态缺失时，自动使用低优先级模态
    """

    def __init__(self,
                 initial_weights: Optional[Dict[str, float]] = None,
                 modality_names: Optional[List[str]] = None,
                 fallback_order: Optional[List[str]] = None,
                 min_confidence_threshold: float = 0.01):
        """
        Args:
            initial_weights: 初始权重
            modality_names: 模态名称
            fallback_order: 降级顺序（从高到低）
            min_confidence_threshold: 最小置信度阈值
        """
        super().__init__(initial_weights, modality_names, min_confidence_threshold)

        if fallback_order is None:
            fallback_order = ['quantitative', 'semantic', 'visual']
        self.fallback_order = fallback_order

    def get_fallback_strategy(self, available_modalities: List[str]) -> Dict:
        """
        获取降级策略

        Args:
            available_modalities: 可用模态列表

        Returns:
            strategy: 降级策略信息
        """
        # 按fallback_order排序可用模态
        sorted_available = [m for m in self.fallback_order if m in available_modalities]

        if not sorted_available:
            return {
                'primary_modality': None,
                'backup_modalities': [],
                'recommendation': 'reject',
                'reason': 'No reliable modality available'
            }

        return {
            'primary_modality': sorted_available[0],
            'backup_modalities': sorted_available[1:] if len(sorted_available) > 1 else [],
            'recommendation': 'proceed' if len(sorted_available) >= 2 else 'caution',
            'reason': f'Using {len(sorted_available)} modalities'
        }

    def fuse_with_confidence(self,
                            risk_scores: Dict[str, torch.Tensor],
                            confidences: Dict[str, torch.Tensor],
                            adjusted_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        带降级策略的融合
        """
        fused_score, fusion_info = super().fuse_with_confidence(
            risk_scores, confidences, adjusted_weights
        )

        # 添加降级策略信息
        available = fusion_info.get('available_modalities', [])
        fallback_strategy = self.get_fallback_strategy(available)
        fusion_info['fallback_strategy'] = fallback_strategy

        return fused_score, fusion_info
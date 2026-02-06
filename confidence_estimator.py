"""
证据置信度估计器
"""
import torch
import torch.nn as nn
from typing import Dict, List


class ConfidenceEstimator(nn.Module):
    """
    多模态证据置信度估计
    考虑可信度、覆盖度、一致性三个维度
    """

    def __init__(self):
        super().__init__()

        # 可学习的权重（如果需要端到端训练）
        self.trustworthiness_weight = nn.Parameter(torch.tensor(0.4))
        self.coverage_weight = nn.Parameter(torch.tensor(0.3))
        self.consistency_weight = nn.Parameter(torch.tensor(0.3))

    def compute_quantitative_confidence(self, features: torch.Tensor) -> Dict:
        """
        计算量化证据的置信度

        Args:
            features: (batch_size, 100) 特征向量

        Returns:
            confidence_dict: 包含置信度和不确定性来源
        """
        batch_size = features.shape[0]

        # 1. 可信度：检查是否有对抗特征
        # 例如：domain_age=0可能表示WHOIS查询失败
        domain_age = features[:, 26]
        registration_length = features[:, 25]
        trustworthiness = 1.0 - 0.3 * (domain_age == 0).float() - \
                          0.2 * (registration_length == 0).float()

        # 2. 覆盖度：检查关键特征是否缺失
        # HTML特征全为0可能表示没有提供HTML
        html_features = features[:, 38:57]
        has_html = (html_features.sum(dim=1) > 0).float()
        coverage = 0.7 + 0.3 * has_html  # 至少0.7（只有URL），有HTML则1.0

        # 3. 一致性：检查规则特征是否冲突
        # 例如：同时触发多个矛盾的规则
        rule_features = features[:, 82:92]
        num_triggered_rules = (rule_features > 0).sum(dim=1).float()
        # 触发过多规则可能表示特征不稳定
        consistency = torch.clamp(1.0 - 0.05 * (num_triggered_rules - 3), 0.5, 1.0)

        # 加权融合
        weights = torch.softmax(torch.stack([
            self.trustworthiness_weight,
            self.coverage_weight,
            self.consistency_weight
        ]), dim=0)

        confidence = weights[0] * trustworthiness + \
                     weights[1] * coverage + \
                     weights[2] * consistency

        # 收集不确定性来源
        uncertainty_sources = []
        for i in range(batch_size):
            sources = []
            if domain_age[i] == 0:
                sources.append("domain_age_unavailable")
            if has_html[i] == 0:
                sources.append("html_content_missing")
            if num_triggered_rules[i] > 5:
                sources.append("excessive_rule_conflicts")
            uncertainty_sources.append(sources)

        return {
            'confidence': confidence,  # (batch_size,)
            'trustworthiness': trustworthiness,
            'coverage': coverage,
            'consistency': consistency,
            'uncertainty_sources': uncertainty_sources
        }

    def compute_visual_confidence(self,
                                  logo_confidence: torch.Tensor,
                                  manipulation_detected: torch.Tensor) -> Dict:
        """
        计算视觉证据的置信度

        Args:
            logo_confidence: (batch_size,) Logo检测置信度 [0, 1]
            manipulation_detected: (batch_size,) 是否检测到manipulation [0/1]

        Returns:
            confidence_dict
        """
        # 如果检测到manipulation，大幅降低置信度
        confidence = logo_confidence * (1.0 - 0.7 * manipulation_detected)

        # 不确定性来源
        batch_size = logo_confidence.shape[0]
        uncertainty_sources = []
        for i in range(batch_size):
            sources = []
            if logo_confidence[i] < 0.5:
                sources.append("logo_unclear")
            if manipulation_detected[i] > 0.5:
                sources.append("visual_manipulation_detected")
            uncertainty_sources.append(sources)

        return {
            'confidence': confidence,
            'logo_confidence': logo_confidence,
            'manipulation_detected': manipulation_detected,
            'uncertainty_sources': uncertainty_sources
        }

    def compute_semantic_confidence(self,
                                    intent_clarity: torch.Tensor,
                                    brand_consistency: torch.Tensor) -> Dict:
        """
        计算语义证据的置信度

        Args:
            intent_clarity: (batch_size,) 意图清晰度 [0, 1]
            brand_consistency: (batch_size,) 品牌一致性 [0, 1]

        Returns:
            confidence_dict
        """
        # 综合意图清晰度和品牌一致性
        confidence = 0.6 * intent_clarity + 0.4 * brand_consistency

        batch_size = intent_clarity.shape[0]
        uncertainty_sources = []
        for i in range(batch_size):
            sources = []
            if intent_clarity[i] < 0.5:
                sources.append("intent_unclear")
            if brand_consistency[i] < 0.5:
                sources.append("brand_inconsistent")
            uncertainty_sources.append(sources)

        return {
            'confidence': confidence,
            'intent_clarity': intent_clarity,
            'brand_consistency': brand_consistency,
            'uncertainty_sources': uncertainty_sources
        }

    def compute_overall_confidence(self,
                                   confidences: Dict[str, torch.Tensor]) -> Dict:
        """
        计算整体置信度（所有模态）

        Args:
            confidences: Dict包含'quantitative', 'visual', 'semantic'的置信度

        Returns:
            overall_confidence_dict
        """
        quant_conf = confidences.get('quantitative', torch.tensor(0.8))
        visual_conf = confidences.get('visual', torch.tensor(0.8))
        semantic_conf = confidences.get('semantic', torch.tensor(0.8))

        # 使用最小值作为整体置信度（保守估计）
        overall_confidence = torch.min(torch.stack([
            quant_conf, visual_conf, semantic_conf
        ], dim=0), dim=0)[0]

        # 计算不确定性（1 - confidence）
        uncertainty = 1.0 - overall_confidence

        return {
            'overall_confidence': overall_confidence,
            'uncertainty': uncertainty,
            'quantitative_confidence': quant_conf,
            'visual_confidence': visual_conf,
            'semantic_confidence': semantic_conf
        }
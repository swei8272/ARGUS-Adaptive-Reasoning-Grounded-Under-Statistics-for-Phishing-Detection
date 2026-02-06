"""
冲突检测器
"""
import torch
from typing import Dict, List, Tuple


class ConflictDetector:
    """
    跨模态冲突检测
    """

    def __init__(self):
        self.conflict_types = [
            'brand_domain_mismatch',
            'visual_semantic_mismatch',
            'new_domain_old_brand',
            'credential_harvesting_external_action'
        ]

    def detect_conflicts(self, evidence: Dict) -> Dict[str, bool]:
        """
        检测各种冲突

        Args:
            evidence: Dict包含quantitative, visual, semantic等证据

        Returns:
            conflicts: Dict[conflict_type, is_detected]
        """
        conflicts = {}

        # 1. 品牌-域名不匹配
        conflicts['brand_domain_mismatch'] = self._detect_brand_domain_mismatch(evidence)

        # 2. 视觉-语义不匹配
        conflicts['visual_semantic_mismatch'] = self._detect_visual_semantic_mismatch(evidence)

        # 3. 新域名宣称老品牌
        conflicts['new_domain_old_brand'] = self._detect_new_domain_old_brand(evidence)

        # 4. 凭据采集+外部action（高危）
        conflicts['credential_harvesting_external_action'] = \
            self._detect_credential_external_action(evidence)

        return conflicts

    def _detect_brand_domain_mismatch(self, evidence: Dict) -> bool:
        """
        检测品牌和域名是否不匹配

        逻辑：
        - 视觉分支识别出某个品牌（高置信度）
        - 但域名不属于该品牌
        """
        visual = evidence.get('visual', {})
        quantitative = evidence.get('quantitative', {})

        # 视觉识别的品牌
        brand_topk = visual.get('brand_topk', [])
        if not brand_topk or len(brand_topk) == 0:
            return False

        top_brand, top_similarity = brand_topk[0]

        # 如果相似度很高
        if top_similarity > 0.8:
            # 检查域名是否匹配
            domain = quantitative.get('domain', '').lower()
            brand_lower = top_brand.lower()

            # 简单的匹配检查（实际应该更复杂）
            if brand_lower not in domain and domain not in brand_lower:
                return True

        return False

    def _detect_visual_semantic_mismatch(self, evidence: Dict) -> bool:
        """
        检测视觉和语义是否不匹配

        逻辑：
        - 视觉模糊/不清晰
        - 语义分支识别出明确的品牌宣称
        """
        visual = evidence.get('visual', {})
        semantic = evidence.get('semantic', {})

        logo_confidence = visual.get('logo_confidence', 1.0)
        manipulation_detected = visual.get('manipulation_detected', False)

        brand_claims = semantic.get('brand_claims', [])

        # 视觉证据弱 + 语义有明确品牌宣称
        if (logo_confidence < 0.5 or manipulation_detected) and len(brand_claims) > 0:
            return True

        return False

    def _detect_new_domain_old_brand(self, evidence: Dict) -> bool:
        """
        检测新域名宣称老品牌

        逻辑：
        - 域名注册时间很短
        - 宣称的是知名品牌
        """
        quantitative = evidence.get('quantitative', {})
        semantic = evidence.get('semantic', {})
        visual = evidence.get('visual', {})

        # 检查域名年龄
        features = quantitative.get('features', torch.zeros(100))
        very_new_domain = features[93] if len(features) > 93 else 0.0  # 索引93

        # 检查是否宣称知名品牌
        brand_claims = semantic.get('brand_claims', [])
        brand_topk = visual.get('brand_topk', [])

        well_known_brands = ['paypal', 'google', 'amazon', 'microsoft', 'apple',
                             'facebook', 'chase', 'wellsfargo', 'bankofamerica']

        claimed_well_known = any(
            any(brand.lower() in claim.lower() for brand in well_known_brands)
            for claim in brand_claims
        )

        detected_well_known = any(
            any(brand.lower() in detected_brand.lower() for brand in well_known_brands)
            for detected_brand, _ in brand_topk
        )

        if very_new_domain > 0.5 and (claimed_well_known or detected_well_known):
            return True

        return False

    def _detect_credential_external_action(self, evidence: Dict) -> bool:
        """
        检测凭据采集+外部提交（最高危）

        逻辑：
        - 有密码输入框
        - 表单提交到外部域名
        """
        quantitative = evidence.get('quantitative', {})
        features = quantitative.get('features', torch.zeros(100))

        if len(features) < 44:
            return False

        num_password_fields = features[40]  # 索引40
        form_action_external = features[43]  # 索引43

        if num_password_fields > 0 and form_action_external > 0.5:
            return True

        return False
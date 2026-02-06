"""
H-FCPG提示词生成器 - 层次化特征条件提示生成
"""
from typing import Dict, List, Optional
import json


class HierarchicalPromptGenerator:
    """
    层次化提示词生成器
    三层结构：系统约束 + 证据摘要 + 检查任务
    """

    def __init__(self, task_names: List[str]):
        """
        Args:
            task_names: 所有任务名称列表
        """
        self.task_names = task_names
        self.task_templates = self._initialize_task_templates()

    def _initialize_task_templates(self) -> Dict[str, str]:
        """初始化任务模板"""
        templates = {
            'domain_brand_consistency':
                "Check if the registered domain matches the claimed brand. "
                "If inconsistent, mark as HIGH RISK and downweight visual evidence.",

            'high_entropy_subdomain_check':
                "Analyze if high-entropy subdomains indicate DGA (Domain Generation Algorithm) "
                "or brand-in-subdomain masquerading.",

            'ip_punycode_risk_check':
                "Check for IP address usage or Punycode encoding. "
                "Explain the specific risk indicators detected.",

            'domain_age_verification':
                "Verify domain age and registration period. "
                "Flag 'new domain + sensitive action' as HIGH RISK.",

            'form_action_consistency':
                "Check if form action domains match the page domain. "
                "Flag external submissions as suspicious.",

            'credential_harvesting_check':
                "Analyze if password fields exist with external form actions. "
                "Assess credential harvesting risk.",

            'external_resource_check':
                "Check if external scripts, iframes, or favicons load from unrelated domains. "
                "Assess injection or tracking risks.",

            'redirect_chain_check':
                "Analyze redirect chains and null links. "
                "Assess if used to evade static detection.",

            'interaction_evidence_reinforcement':
                "Prioritize interaction features (combined evidence) over individual features. "
                "Focus on multi-signal consistency.",

            'login_external_link_priority':
                "When login/password fields AND high external link ratio detected, "
                "escalate risk priority and explain the attack chain.",

            'typosquatting_detection':
                "Identify the target brand being mimicked. "
                "Point out character substitutions, omissions, or insertions.",

            'homograph_attack_detection':
                "Identify if Punycode or character set confusion is used. "
                "Highlight suspicious character positions.",

            'phishing_kit_pattern':
                "Check for typical phishing kit structures or concentrated external resource loading.",

            'credential_form_external_action':
                "Align input fields, submission target, and page intent. "
                "Flag misalignment as HIGH RISK.",

            'complex_subdomain_structure':
                "Analyze if subdomain nesting is used for brand masquerading, "
                "especially when brand appears in subdomain rather than registered domain.",

            'short_high_entropy_domain':
                "Analyze short domains with high entropy. "
                "Assess randomness indicators.",

            'temporal_consistency_check':
                "Explain why attackers prefer short-term registration and new domains. "
                "Use temporal evidence as key decision factor."
        }
        return templates

    def generate_prompt(self,
                        triggered_tasks: List[str],
                        evidence_dict: Dict,
                        conflict_info: Optional[Dict] = None) -> str:
        """
        生成完整的提示词

        Args:
            triggered_tasks: 被触发的任务列表
            evidence_dict: 证据字典（quantitative, visual, semantic）
            conflict_info: 冲突信息（可选）

        Returns:
            prompt: 完整提示词
        """
        # 第一层：系统约束
        system_constraints = self._generate_system_constraints()

        # 第二层：证据摘要
        evidence_digest = self._generate_evidence_digest(evidence_dict)

        # 第三层：检查任务
        inspection_tasks = self._generate_inspection_tasks(triggered_tasks)

        # 第四层：冲突说明（如果有）
        conflict_section = ""
        if conflict_info and conflict_info.get('conflicts_detected'):
            conflict_section = self._generate_conflict_section(conflict_info)

        # 组合
        prompt = f"""{system_constraints}

{evidence_digest}

{inspection_tasks}

{conflict_section}

Based on the evidence and tasks above, provide:
1. Risk assessment (HIGH/MEDIUM/LOW)
2. Confidence level (0-1)
3. Key evidence supporting your decision
4. Uncertainty sources (if any)
5. Recommended action (BLOCK/WARN/ALLOW/MANUAL_REVIEW)

Output in JSON format.
"""

        return prompt

    def _generate_system_constraints(self) -> str:
        """生成系统级约束"""
        return """# SYSTEM CONSTRAINTS

You are a phishing detection expert. Analyze the provided evidence and complete the specified inspection tasks.

**Output Requirements:**
- Use JSON format
- Cite evidence sources explicitly (e.g., "quantitative.domain_entropy=4.2")
- Explain conflicting evidence
- Provide uncertainty estimates

**Risk Priority Rules:**
- Credential harvesting + external action → HIGH RISK (override other signals)
- Brand-domain mismatch → Downweight visual evidence
- New domain + established brand claim → HIGH RISK
"""

    def _generate_evidence_digest(self, evidence_dict: Dict) -> str:
        """生成证据摘要（第二层）"""
        # Quantitative证据
        quant_summary = self._summarize_quantitative(evidence_dict.get('quantitative', {}))

        # Visual证据
        visual_summary = self._summarize_visual(evidence_dict.get('visual', {}))

        # Semantic证据
        semantic_summary = self._summarize_semantic(evidence_dict.get('semantic', {}))

        return f"""# EVIDENCE DIGEST

## Quantitative Evidence
{quant_summary}

## Visual Evidence
{visual_summary}

## Semantic Evidence
{semantic_summary}
"""

    def _summarize_quantitative(self, quant_evidence: Dict) -> str:
        """总结量化证据"""
        features = quant_evidence.get('features', None)
        if features is None:
            return "No quantitative features available."

        # 转换为numpy以便索引
        import torch
        import numpy as np

        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        elif not isinstance(features, np.ndarray):
            return "Invalid feature format."

        # 提取关键特征
        summary_parts = []

        # URL特征
        if len(features) > 26:
            url_len = features[0] if features.ndim > 0 else 0
            domain_entropy = features[17] if len(features) > 17 else 0
            summary_parts.append(f"- URL: length={url_len:.0f}, domain_entropy={domain_entropy:.2f}")

            num_subdomains = features[16] if len(features) > 16 else 0
            suspicious_tld = features[19] if len(features) > 19 else 0
            summary_parts.append(f"- Domain: subdomains={num_subdomains:.0f}, suspicious_tld={suspicious_tld:.0f}")

            domain_age = features[26] if len(features) > 26 else 0
            summary_parts.append(f"- Temporal: domain_age={domain_age:.0f} days")

        # HTML特征
        if len(features) > 56:
            num_forms = features[38] if len(features) > 38 else 0
            num_password = features[40] if len(features) > 40 else 0
            external_action = features[43] if len(features) > 43 else 0
            summary_parts.append(
                f"- Forms: count={num_forms:.0f}, password_fields={num_password:.0f}, external_action={external_action:.0f}")

            external_link_ratio = features[47] if len(features) > 47 else 0
            summary_parts.append(f"- Links: external_ratio={external_link_ratio:.2f}")

        # 规则特征
        if len(features) > 91:
            triggered_rules = []
            rule_names = ['typosquatting', 'homograph', 'phishing_kit', 'credential_harvesting']
            rule_indices = [82, 83, 84, 85]

            for name, idx in zip(rule_names, rule_indices):
                if len(features) > idx and features[idx] > 0.5:
                    triggered_rules.append(name)

            if triggered_rules:
                summary_parts.append(f"- Triggered rules: {', '.join(triggered_rules)}")

        return "\n".join(summary_parts) if summary_parts else "Features available but key indicators not extracted."

    def _summarize_visual(self, visual_evidence: Dict) -> str:
        """总结视觉证据"""
        brand_topk = visual_evidence.get('brand_topk', [])
        logo_conf = visual_evidence.get('logo_confidence', 0.0)
        manipulation = visual_evidence.get('manipulation_detected', False)

        summary = []

        if brand_topk:
            if isinstance(brand_topk, list) and len(brand_topk) > 0:
                top_brands = ", ".join([f"{brand}({sim:.2f})" for brand, sim in brand_topk[:3]])
                summary.append(f"- Brand candidates: {top_brands}")
            else:
                summary.append("- Brand candidates: Invalid format")
        else:
            summary.append("- No brand detected")

        summary.append(f"- Logo confidence: {logo_conf:.2f}")

        if manipulation:
            summary.append("- ⚠️ Visual manipulation detected (blur/low-res/distortion)")

        return "\n".join(summary)

    def _summarize_semantic(self, semantic_evidence: Dict) -> str:
        """总结语义证据"""
        intent = semantic_evidence.get('intent_type', 'unknown')
        sensitive_actions = semantic_evidence.get('sensitive_action_flags', [])
        social_eng = semantic_evidence.get('social_engineering_flags', [])
        brand_claims = semantic_evidence.get('brand_claims', [])

        summary = []
        summary.append(f"- Page intent: {intent}")

        if sensitive_actions:
            if isinstance(sensitive_actions, list):
                summary.append(f"- Sensitive actions: {', '.join(sensitive_actions)}")
            else:
                summary.append(f"- Sensitive actions: {sensitive_actions}")

        if social_eng:
            if isinstance(social_eng, list):
                summary.append(f"- Social engineering signals: {', '.join(social_eng)}")
            else:
                summary.append(f"- Social engineering signals: {social_eng}")

        if brand_claims:
            if isinstance(brand_claims, list):
                summary.append(f"- Brand claims: {', '.join(brand_claims)}")
            else:
                summary.append(f"- Brand claims: {brand_claims}")

        return "\n".join(summary)

    def _generate_inspection_tasks(self, triggered_tasks: List[str]) -> str:
        """生成检查任务（第三层）"""
        if not triggered_tasks:
            return "# INSPECTION TASKS\n\nNo specific tasks triggered. Perform general phishing assessment."

        task_section = "# INSPECTION TASKS\n\n"
        task_section += f"**{len(triggered_tasks)} tasks triggered (prioritized by risk):**\n\n"

        for i, task_name in enumerate(triggered_tasks, 1):
            template = self.task_templates.get(task_name, f"Analyze {task_name}")
            task_section += f"**Task {i}: {task_name}**\n{template}\n\n"

        return task_section

    def _generate_conflict_section(self, conflict_info: Dict) -> str:
        """生成冲突说明"""
        conflicts = conflict_info.get('conflicts_detected', [])
        resolution_log = conflict_info.get('resolution_log', [])

        if not conflicts:
            return ""

        section = "# CONFLICT RESOLUTION\n\n"
        section += f"**Detected conflicts:** {', '.join(conflicts)}\n\n"
        section += "**Resolution applied:**\n"
        for log_entry in resolution_log:
            section += f"- {log_entry}\n"

        section += "\n**Note:** Consider these conflict resolutions in your final assessment.\n"

        return section

    def generate_batch_prompts(self,
                               triggered_tasks_list: List[List[str]],
                               evidence_list: List[Dict],
                               conflict_info_list: Optional[List[Dict]] = None) -> List[str]:
        """
        批量生成提示词

        Args:
            triggered_tasks_list: List of triggered tasks for each sample
            evidence_list: List of evidence dicts
            conflict_info_list: List of conflict info dicts (optional)

        Returns:
            prompts: List of prompts
        """
        if conflict_info_list is None:
            conflict_info_list = [None] * len(triggered_tasks_list)

        prompts = []
        for tasks, evidence, conflicts in zip(triggered_tasks_list, evidence_list, conflict_info_list):
            prompt = self.generate_prompt(tasks, evidence, conflicts)
            prompts.append(prompt)

        return prompts

    def estimate_token_count(self, prompt: str) -> int:
        """
        粗略估计token数量

        Args:
            prompt: 提示词文本

        Returns:
            estimated_tokens: 估计的token数
        """
        # 粗略估计：英文约4个字符=1个token
        return len(prompt) // 4

    def get_prompt_statistics(self, prompts: List[str]) -> Dict:
        """
        获取一批提示词的统计信息

        Args:
            prompts: 提示词列表

        Returns:
            stats: 统计信息
        """
        if not prompts:
            return {'error': 'No prompts provided'}

        token_counts = [self.estimate_token_count(p) for p in prompts]
        lengths = [len(p) for p in prompts]

        return {
            'num_prompts': len(prompts),
            'avg_length_chars': sum(lengths) / len(lengths),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'max_tokens': max(token_counts),
            'min_tokens': min(token_counts),
            'total_tokens': sum(token_counts)
        }
"""
冲突消解器 - 带优先级的层次化处理
"""
import torch
from typing import Dict, List, Tuple  # ← 添加 Tuple
from ARGUS.utils.config import Config


class ConflictResolver:
    """
    冲突消解器
    按优先级处理多个冲突
    """

    def __init__(self, conflict_priority: Dict = None):
        """
        Args:
            conflict_priority: 冲突优先级配置（默认使用Config中的）
        """
        self.priority_config = conflict_priority or Config.CONFLICT_PRIORITY

    def resolve_conflicts(self,
                         conflicts: Dict[str, bool],
                         evidence: Dict,
                         current_weights: Dict[str, float]) -> Dict:
        """
        按优先级处理冲突

        Args:
            conflicts: Dict[conflict_type, is_detected]
            evidence: 证据字典
            current_weights: 当前的模态权重

        Returns:
            resolution_result: 包含调整后的权重、额外任务、解释等
        """
        # 筛选出实际发生的冲突
        detected_conflicts = {k: v for k, v in conflicts.items() if v}

        if not detected_conflicts:
            return {
                'adjusted_weights': current_weights,
                'additional_tasks': [],
                'risk_adjustment': 0.0,
                'resolution_log': []
            }

        # 按优先级排序
        sorted_conflicts = self._sort_by_priority(detected_conflicts)

        # 初始化结果
        adjusted_weights = current_weights.copy()
        additional_tasks = []
        risk_adjustment = 0.0
        resolution_log = []

        # 逐个处理冲突
        for conflict_type, priority_info in sorted_conflicts:
            action = priority_info['action']

            # 执行相应的消解策略
            if action == 'force_high_risk':
                risk_adjustment = 1.0  # 强制高风险
                resolution_log.append(f"CRITICAL: {conflict_type} → Force high risk")

                # 如果有override标志，停止处理后续冲突
                if priority_info.get('override', False):
                    resolution_log.append(f"Override: Stop further conflict resolution")
                    break

            elif action == 'downweight_visual':
                adjusted_weights['visual'] *= 0.3  # 视觉权重降至30%
                resolution_log.append(f"{conflict_type} → Downweight visual to {adjusted_weights['visual']:.2f}")

            elif action == 'increase_uncertainty':
                risk_adjustment += 0.2  # 增加不确定性
                resolution_log.append(f"{conflict_type} → Increase uncertainty")

            elif action == 'raise_suspicion':
                risk_adjustment += 0.15
                resolution_log.append(f"{conflict_type} → Raise suspicion")

            # 添加额外任务
            if 'add_tasks' in priority_info:
                additional_tasks.extend(priority_info['add_tasks'])
                resolution_log.append(f"{conflict_type} → Add tasks: {priority_info['add_tasks']}")

        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}

        return {
            'adjusted_weights': adjusted_weights,
            'additional_tasks': list(set(additional_tasks)),  # 去重
            'risk_adjustment': min(risk_adjustment, 1.0),  # 最多+1.0
            'resolution_log': resolution_log,
            'conflicts_detected': list(detected_conflicts.keys())
        }

    def _sort_by_priority(self, conflicts: Dict[str, bool]) -> List[Tuple[str, Dict]]:
        """
        按优先级排序冲突

        Args:
            conflicts: 检测到的冲突

        Returns:
            sorted_conflicts: [(conflict_type, priority_info), ...]
        """
        conflict_list = []
        for conflict_type in conflicts.keys():
            if conflict_type in self.priority_config:
                priority_info = self.priority_config[conflict_type]
                conflict_list.append((conflict_type, priority_info))

        # 按priority字段排序（数字越小优先级越高）
        conflict_list.sort(key=lambda x: x[1]['priority'])

        return conflict_list
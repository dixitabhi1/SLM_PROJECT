"""
Matching SLM (v2 Architecture - Convergence Stage)
Performs within-color agent-task matching, evaluates the loop-back condition, and determines execution mode.
"""

from typing import Dict, List, Any, Optional

class MatchingSLM:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def match_task_and_evaluate_loop(
        self,
        task: Dict[str, Any],
        task_color_info: Dict[str, Any],
        agent_profiles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates task against agent pool within color classes,
        determines parallel/series compatibility, and evaluates the bounded loop condition.
        """
        task_id = task["id"]
        current_depth = task.get("depth", 0)
        spans_multiple = task_color_info["spans_multiple_colors"]
        depth_limit_reached = current_depth >= self.max_depth

        # 1. Evaluate Loop-Back Condition
        if spans_multiple and not depth_limit_reached:
            action = "LOOP_BACK_TO_DECOMPOSER"
            assigned_agent = None
            assigned_team = []
            collaboration_mode = False
        else:
            action = "FORWARD_TO_SCHEDULING"
            if spans_multiple and depth_limit_reached:
                # Terminal multi-color state -> Multi-agent collaboration team
                collaboration_mode = True
                assigned_agent = None
                assigned_team = task_color_info["active_domains"]
            else:
                # Single-color match
                collaboration_mode = False
                assigned_agent = task_color_info["dominant_domain"]
                assigned_team = [assigned_agent]

        # 2. Check parallel vs series constraint from dependencies
        dependencies = task.get("dependencies", [])
        execution_constraint = "SERIES" if len(dependencies) > 0 else "PARALLEL_CANDIDATE"

        return {
            "task_id": task_id,
            "current_depth": current_depth,
            "action": action,
            "spans_multiple_colors": spans_multiple,
            "depth_limit_reached": depth_limit_reached,
            "collaboration_mode": collaboration_mode,
            "assigned_agent": assigned_agent,
            "assigned_team": assigned_team,
            "dominant_color": task_color_info["dominant_color"],
            "active_colors": task_color_info["active_colors"],
            "execution_constraint": execution_constraint
        }


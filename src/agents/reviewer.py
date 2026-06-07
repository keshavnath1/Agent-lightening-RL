from __future__ import annotations

import json
from pathlib import Path
from src.agents.base import BaseAgent, AgentContext
from src.telemetry.schema import AgentStep
from src.governance.guardrails import validate_workflow_artifacts


class ReviewerCriticAgent(BaseAgent):
    name = 'ReviewerCriticAgent'

    def step(self, context: AgentContext) -> AgentStep:
        review = validate_workflow_artifacts(context)
        task_id = context.task['task_id']
        review_path = Path('artifacts') / task_id / 'review_report.json'
        review_path.write_text(json.dumps(review, indent=2), encoding='utf-8')
        context.artifacts['review_report_json'] = str(review_path)
        context.tool_outputs['review'] = review
        action = 'approve_workflow_artifacts' if review['approved'] else 'flag_workflow_artifact_issues'
        return AgentStep(
            agent_name=self.name,
            action=action,
            reasoning_summary='Reviewer validated artifact existence, model output, tracking completeness, sandbox mode, and raw-data leakage markers.',
            tool_calls=[self.tool_success('GuardrailValidator', {'task_id': task_id, 'approved': review['approved']}, str(review_path))],
        )

from __future__ import annotations

import json
from pathlib import Path
from src.agents.base import BaseAgent, AgentContext
from src.telemetry.schema import AgentStep


class GradientBoostingSpecialistAgent(BaseAgent):
    name = 'GradientBoostingSpecialistAgent'

    def step(self, context: AgentContext) -> AgentStep:
        schema_path = Path(context.artifacts['schema_metadata_json'])
        profile_path = Path(context.artifacts['profile_summary_json'])
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        profile = json.loads(profile_path.read_text(encoding='utf-8'))
        row_count = int(schema['row_count'])
        high_cardinality_columns = [
            c['name'] for c in profile.get('columns', [])
            if c.get('name') != schema.get('target_column') and c.get('is_high_cardinality')
        ]
        high_cardinality = bool(high_cardinality_columns)
        selected_model = 'LightGBM' if row_count > 1_000_000 or high_cardinality else 'XGBoost'
        plan_path = schema_path.parent / 'gbm_benchmark_plan.json'
        plan = {
            'selected_primary_model': selected_model,
            'benchmark_candidates': ['XGBoost', 'LightGBM'],
            'cv_strategy': 'train_validation_split_for_mvp; stratified_5_fold_for_full_run',
            'metric': context.task.get('metric', 'roc_auc'),
            'row_count': row_count,
            'high_cardinality_columns': high_cardinality_columns,
            'selection_rationale': 'LightGBM is preferred for very large or high-cardinality tasks; otherwise XGBoost is the primary candidate, with both benchmarked when dependencies are available.',
        }
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
        context.artifacts['gbm_benchmark_plan'] = str(plan_path)
        return AgentStep(
            agent_name=self.name,
            action='select_gbm_benchmark_strategy_from_profile',
            reasoning_summary=f'Selected {selected_model} as primary model using row count and profile-derived high-cardinality evidence.',
            tool_calls=[self.tool_success('ProfileAndSchemaMetadataReader', {'schema_path': str(schema_path), 'profile_path': str(profile_path)}, str(plan_path))],
        )

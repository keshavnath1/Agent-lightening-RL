from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class PolicyEndpointConfig:
    policy_version: str
    base_url: str
    model_name: str
    api_key: str | None = None


_PROFILE_ENV_PREFIX = {
    'baseline': 'BASELINE_POLICY',
    'baseline_train': 'BASELINE_POLICY',
    'v2': 'V2_POLICY',
    'policy_v2': 'V2_POLICY',
    'rl_tuned': 'TUNED_POLICY',
    'tuned': 'TUNED_POLICY',
    'agent_lightning_policy': 'TUNED_POLICY',
}


def _normalize_policy_version(policy_version: str | None) -> str:
    return (policy_version or os.getenv('POLICY_VERSION') or 'baseline').strip().lower()


def _env_profile_prefix(policy_version: str) -> str:
    normalized = _normalize_policy_version(policy_version)
    if normalized in _PROFILE_ENV_PREFIX:
        return _PROFILE_ENV_PREFIX[normalized]
    safe = ''.join(ch if ch.isalnum() else '_' for ch in normalized).upper()
    return f'{safe}_POLICY'


def resolve_policy_endpoint(
    policy_version: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> PolicyEndpointConfig:
    """Resolve an OpenAI-compatible GPU endpoint for a named benchmark policy.

    The resolver is intentionally strict. It does not invent localhost defaults because
    benchmark reports must reflect the intended GPU endpoint, for example baseline vs
    V2 vs tuned. Use BASELINE_POLICY_URL, V2_POLICY_URL, or TUNED_POLICY_URL, with
    matching *_MODEL and optional *_API_KEY values.
    """
    normalized = _normalize_policy_version(policy_version)
    prefix = _env_profile_prefix(normalized)
    resolved_url = base_url or os.getenv(f'{prefix}_URL')
    resolved_model = model_name or os.getenv(f'{prefix}_MODEL') or os.getenv('MODEL_NAME')
    resolved_api_key = api_key or os.getenv(f'{prefix}_API_KEY') or os.getenv('POLICY_API_KEY')
    if not resolved_url:
        raise RuntimeError(
            f'Missing policy endpoint URL for policy_version={normalized}. '
            f'Set {prefix}_URL, such as BASELINE_POLICY_URL, V2_POLICY_URL, or TUNED_POLICY_URL.'
        )
    if not resolved_model:
        raise RuntimeError(
            f'Missing model name for policy_version={normalized}. Set {prefix}_MODEL or MODEL_NAME.'
        )
    return PolicyEndpointConfig(
        policy_version=normalized,
        base_url=resolved_url,
        model_name=resolved_model,
        api_key=resolved_api_key,
    )


class PolicyClient:
    def __init__(
        self,
        policy_version: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        self.config = resolve_policy_endpoint(policy_version, base_url, model_name, api_key)

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        payload = {'model': self.model_name, 'messages': messages, 'temperature': temperature}
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']

    def decision(self, task: dict[str, Any], policy_version: str | None = None) -> dict[str, Any]:
        version = policy_version or self.config.policy_version
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a policy model for a self-improving ML-agent benchmark. '
                    'Return compact JSON with plan, risk_controls, and expected_artifacts. '
                    'Do not request or expose raw dataset rows.'
                ),
            },
            {
                'role': 'user',
                'content': 'TASK_JSON:\n' + str(task) + f'\nPOLICY_VERSION: {version}',
            },
        ]
        content = self.chat(messages)
        return {
            'policy_version': version,
            'endpoint_url_redacted': self.base_url.split('?')[0],
            'model_name': self.model_name,
            'content': content,
        }

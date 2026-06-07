.PHONY: validate-monorepo test-monorepo validate-mcp validate-ruler validate-agent-lightning-strict dashboard track-a track-b

validate-monorepo:
	python scripts/validate/validate_monorepo_phase1.py

test-monorepo:
	PYTHONPATH=.:packages/contracts/src:packages/ml_tools/src:packages/rewards/src:packages/lightning_bridge/src:packages/mcp_client_bridge/src:services/project_mcp_server/src:apps/dashboard/src:apps/rollout_worker/src:apps/ruler_scorer/src:apps/trainer/src pytest -q tests/contracts tests/tools tests/rewards tests/mcp tests/training tests/integration

validate-mcp:
	python scripts/dev_validate_official_mcp_server.py

validate-ruler:
	python scripts/dev_validate_official_art_ruler.py

validate-agent-lightning-strict:
	python scripts/validate/validate_agent_lightning_strict.py

dashboard:
	streamlit run scripts/demo_dashboard.py

track-a:
	bash scripts/cpu/run_05c_agent_lightning_official.sh

track-b:
	bash scripts/cpu/run_08_benchmark_policy_endpoints.sh

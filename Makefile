.PHONY: hooks test

hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push

test:
	.venv/bin/python tests/test_accuracy.py
	.venv/bin/python tests/test_medical_stats.py
	.venv/bin/python tests/test_analysis_outputs.py
	.venv/bin/python tests/test_baseline_validation.py
	.venv/bin/python tests/test_stage_failures.py
	.venv/bin/python tests/test_medical_shell.py

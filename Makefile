.PHONY: hooks test run

run:
	.venv/bin/python -m streamlit run app/streamlit_app.py --server.address=127.0.0.1

hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push

test:
	.venv/bin/python tests/test_accuracy.py
	.venv/bin/python tests/test_medical_stats.py
	.venv/bin/python tests/test_analysis_outputs.py
	.venv/bin/python tests/test_baseline_validation.py
	.venv/bin/python tests/test_stage_failures.py
	.venv/bin/python tests/test_llm_token_limits.py
	.venv/bin/python tests/test_resume_selection.py
	.venv/bin/python tests/test_saved_run_security.py
	.venv/bin/python tests/test_streamlit_app.py

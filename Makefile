.PHONY: hooks test

hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push

test:
	.venv/bin/python tests/test_accuracy.py

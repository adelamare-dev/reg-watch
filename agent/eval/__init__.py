"""Golden-dataset harness: schema, loading and validation for the offline evaluation set.

Depends on `graph`/`rag`/`mcp_server` in only one direction — this package
consumes their state shapes to build assertions, never the reverse
(see `tests/test_eval_architecture.py`).
"""

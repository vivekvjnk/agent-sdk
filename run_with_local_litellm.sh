#!/usr/bin/env bash
export PYTHONPATH=/home/pst/Documents/crazy_orca/litellm:$PYTHONPATH
uv run python "$@"

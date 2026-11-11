"""Backward-compatible entry point; use src/infer.py.

Earlier commands and experiment notes call this file for both zero-shot and
adapter inference, so it stays as a thin alias.
"""

from __future__ import annotations

try:
    from .infer import main, parse_args, positive_integer, project_root
except ImportError:  # python src/zero_shot.py
    from infer import main, parse_args, positive_integer, project_root

__all__ = ["main", "parse_args", "positive_integer", "project_root"]


if __name__ == "__main__":
    main()

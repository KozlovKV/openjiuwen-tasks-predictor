"""Entry point: python run_agent.py --tasks-dir DIR [--out FILE]."""
from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import sys
import tempfile
from pathlib import Path


def quiet_openjiuwen() -> None:
    """Keep openJiuwen off the disk: by default it logs to ./logs, which is read-only."""
    from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG
    from openjiuwen.core.common.logging.log_config import configure_log_config

    config = copy.deepcopy(DEFAULT_INNER_LOG_CONFIG)
    config["level"] = "CRITICAL"
    config["propagate"] = False
    config["log_path"] = str(Path(tempfile.gettempdir()) / "jiuwen-logs")
    for key in ("output", "interface_output", "performance_output"):
        config[key] = ["console"]
    configure_log_config(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("harness").setLevel(logging.INFO)
    quiet_openjiuwen()
    from harness.config import load_settings
    from harness.runner import run

    asyncio.run(run(load_settings(args.tasks_dir, args.out)))


if __name__ == "__main__":
    main()

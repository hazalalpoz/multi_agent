from task_blocks.config import get_settings
from task_blocks.graph import build_graph

__all__ = ["build_graph", "get_settings", "main"]

__version__ = "0.1.0"


def main() -> None:
    from task_blocks.cli import main as cli_main

    cli_main()

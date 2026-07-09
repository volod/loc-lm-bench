"""CLI entry point (Typer). Implementation lives under llb.cli."""

from llb.cli import app


def main() -> None:
    from llb.core.runtime import run_typer

    run_typer(app)


if __name__ == "__main__":
    main()

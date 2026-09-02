"""Command-line entry point for generating or checking artifact contract exports."""

import argparse
import logging

from llb.artifacts.generation import check_exports, write_exports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh generated contract exports")
    args = parser.parse_args()
    if args.write:
        for path in write_exports():
            logger.info("[artifact-contracts] wrote %s", path)
    problems = check_exports()
    if problems:
        for problem in problems:
            logger.error("[artifact-contracts] ERROR: %s", problem)
        raise SystemExit(1)
    logger.info("[artifact-contracts] OK: registry, schemas, catalog, and ODCS pin are current")


if __name__ == "__main__":
    main()

"""Module entry point for ``python -m llb.goldset.verify``."""

import sys

from llb.core.runtime import run
from llb.goldset.verify.cli import main

sys.exit(run(main))

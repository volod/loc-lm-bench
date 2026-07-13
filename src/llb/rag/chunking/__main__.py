"""`python -m llb.rag.chunking` entry point -- delegates to the build CLI (also `make build-rag-store`)."""

import sys

from llb.core.runtime import run
from llb.rag.chunking.build import main

if __name__ == "__main__":
    sys.exit(run(main))

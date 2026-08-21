"""Which transformers major a candidate's repository code targets, and how a run reaches it.

Four roster candidates an operator would genuinely shortlist -- two encoders
(`Alibaba-NLP/gte-multilingual-base`, `jinaai/jina-embeddings-v3`) and two rerankers
(`Alibaba-NLP/gte-multilingual-reranker-base`, `jinaai/jina-reranker-v2-base-multilingual`) --
ship their forward pass as repository code written against the transformers 4.x API. The repo pins
transformers 5.x for the SHIPPED path, and on that stack all four are unusable: two raise at load,
and two load and return numbers that do not reproduce their own model card. That is a PACKAGING
fact, not a quality one, and a bake-off that silently drops those rows publishes a ranking of
"whose publisher tracked our pin".

So the stack a candidate needs is DECLARED beside its input convention, and the run states which
stack it is on:

  - a candidate with no declared requirement runs anywhere;
  - a candidate that declares `REQUIRED_TRANSFORMERS_MAJOR_LEGACY` runs only on the legacy pass --
    a separate virtualenv holding the `[encoders-legacy]` extra, which pins `transformers<5` beside
    the same sentence-transformers and torch. On the pinned stack it is SKIPPED with the pin it
    would need, rather than failing the run or vanishing from the table.

The legacy pass is a second SCORING environment, never the shipped one: nothing in `src/` imports
it, the repo-wide pin is unchanged, and the two passes are comparable only because the load
precision is declared (`llb.rag.encoders.precision`) rather than inherited from each checkpoint.

Pure and dependency-free: the installed version is read from package metadata, never by importing
transformers.
"""

from importlib.metadata import PackageNotFoundError, version

# The transformers major the repo pins for the shipped path.
PINNED_TRANSFORMERS_MAJOR = 5

# The major the four remote-code candidates' repository code targets.
REQUIRED_TRANSFORMERS_MAJOR_LEGACY = 4

# The optional extra that provides it, and the make targets that run a scoring pass inside it.
LEGACY_EXTRA = "encoders-legacy"
LEGACY_EMBEDDING_TARGET = "make compare-embeddings-legacy"
LEGACY_RERANK_TARGET = "make compare-rerankers-legacy"


def installed_transformers_version() -> str | None:
    """The transformers version in this interpreter, or None when it is not installed."""
    try:
        return version("transformers")
    except PackageNotFoundError:
        return None


def major_of(package_version: str | None) -> int | None:
    """The major component of a version string (None when it is absent or unparseable)."""
    if not package_version:
        return None
    head = package_version.strip().split(".", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def installed_transformers_major() -> int | None:
    """The transformers major this interpreter would load a candidate with."""
    return major_of(installed_transformers_version())


def legacy_pass_hint(subject: str) -> str:
    """The one line an operator acts on when a candidate needs the legacy transformers pass."""
    target = LEGACY_EMBEDDING_TARGET if subject == "an encoder" else LEGACY_RERANK_TARGET
    return (
        f"repository code targets transformers "
        f"{REQUIRED_TRANSFORMERS_MAJOR_LEGACY}.x; the pinned stack is "
        f"{PINNED_TRANSFORMERS_MAJOR}.x. Score it in the legacy pass: `{target}` "
        f"(the [{LEGACY_EXTRA}] extra in its own virtualenv)."
    )

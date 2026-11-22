"""Resolution of one page's block ledger into the text both renderings read."""

from llb.prep.book.resolve.apply import resolve_page
from llb.prep.book.resolve.primary import PrimaryChoice, choice_from_params

__all__ = ["PrimaryChoice", "choice_from_params", "resolve_page"]

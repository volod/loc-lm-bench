"""A dependency-free Turtle reader/writer for the committed axiom set.

The axiom file is the reviewable artifact -- someone who reads OWL but not this codebase must be
able to open it -- so it is real Turtle, not JSON wearing a `.ttl` suffix. The SHIPPED checker
still adds no runtime dependency: this module reads the subset of Turtle the axiom set uses
(prefix directives, predicate-object lists, string/integer literals, anonymous blank nodes, and
collections) into flat triples, and `rdf.py` interprets those triples as axioms.

`rdflib` is never imported here. It appears only in `crosscheck.py`, behind the optional
`[ontology]` extra, as a reasoner cross-check that has no say in the answer path.
"""

import re
from dataclasses import dataclass

RDF_FIRST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
RDF_REST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
RDF_NIL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"


@dataclass(frozen=True)
class Iri:
    """An absolute IRI (prefixed names are expanded at parse time)."""

    value: str


@dataclass(frozen=True)
class Literal:
    """A string or typed literal."""

    value: str
    datatype: str | None = None


@dataclass(frozen=True)
class Bnode:
    """A blank node; anonymous nodes get generated, document-stable labels."""

    label: str


Term = Iri | Literal | Bnode
Triple = tuple[Term, Iri, Term]


class TurtleError(ValueError):
    """The axiom file is not Turtle this reader understands."""


_TOKEN = re.compile(
    r"""(?P<ws>\s+)
      | (?P<comment>\#[^\n]*)
      | (?P<directive>@prefix|@base)
      | (?P<iri><[^<>"{}|^`\\\s]*>)
      | (?P<string>"(?:\\.|[^"\\])*")
      | (?P<datatype>\^\^)
      | (?P<lang>@[A-Za-z][A-Za-z0-9-]*)
      | (?P<number>[+-]?\d+(?:\.\d+)?)
      | (?P<punct>[;,.\[\]()])
      | (?P<bnode>_:[^\s;,.\[\]()]+)
      | (?P<name>[^\s;,.\[\]()<>"^@]+)""",
    re.VERBOSE,
)
_SKIP = frozenset({"ws", "comment"})
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}


def _unescape(raw: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), raw)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise TurtleError(f"unreadable character at offset {pos}: {text[pos : pos + 20]!r}")
        pos = match.end()
        kind = match.lastgroup or ""
        if kind not in _SKIP:
            tokens.append((kind, match.group()))
    return tokens


class _Parser:
    """Recursive-descent reader over the token stream; emits flat triples."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._bnodes = 0
        self.prefixes: dict[str, str] = {}
        self.triples: list[Triple] = []

    # --- token helpers ---
    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise TurtleError("unexpected end of axiom file")
        self._pos += 1
        return token

    def _expect(self, value: str) -> None:
        kind, text = self._next()
        if text != value:
            raise TurtleError(f"expected {value!r}, found {text!r} ({kind})")

    def _fresh_bnode(self) -> Bnode:
        self._bnodes += 1
        return Bnode(f"b{self._bnodes}")

    # --- grammar ---
    def parse(self) -> None:
        while self._peek() is not None:
            kind, text = self._peek()  # type: ignore[misc]
            if kind == "directive":
                self._directive()
            else:
                self._statement()

    def _directive(self) -> None:
        _, name = self._next()
        if name == "@base":  # accepted and ignored: every axiom IRI is written absolute-resolvable
            self._next()
            self._expect(".")
            return
        _, prefix = self._next()
        _, iri = self._next()
        self.prefixes[prefix.rstrip(":")] = iri.strip("<>")
        self._expect(".")

    def _statement(self) -> None:
        subject = self._term()
        self._predicate_object_list(subject)
        self._expect(".")

    def _predicate_object_list(self, subject: Term) -> None:
        while True:
            predicate = self._predicate()
            self._object_list(subject, predicate)
            if self._peek() == ("punct", ";"):
                self._next()
                if self._peek() in (("punct", "."), ("punct", "]")):
                    return
                continue
            return

    def _predicate(self) -> Iri:
        term = self._term()
        if not isinstance(term, Iri):
            raise TurtleError(f"predicate must be an IRI, found {term!r}")
        return term

    def _object_list(self, subject: Term, predicate: Iri) -> None:
        while True:
            self.triples.append((subject, predicate, self._term()))
            if self._peek() == ("punct", ","):
                self._next()
                continue
            return

    def _term(self) -> Term:
        kind, text = self._next()
        if kind == "iri":
            return Iri(text[1:-1])
        if kind == "bnode":
            return Bnode(text[2:])
        if kind == "string":
            return self._literal(_unescape(text[1:-1]))
        if kind == "number":
            return Literal(text, XSD_INTEGER if "." not in text else None)
        if kind == "name":
            return Iri(RDF_TYPE) if text == "a" else Iri(self._expand(text))
        if text == "[":
            return self._anonymous()
        if text == "(":
            return self._collection()
        raise TurtleError(f"unexpected token {text!r} ({kind})")

    def _literal(self, value: str) -> Literal:
        if self._peek() == ("datatype", "^^"):
            self._next()
            term = self._term()
            if not isinstance(term, Iri):
                raise TurtleError("literal datatype must be an IRI")
            return Literal(value, term.value)
        if self._peek() is not None and self._peek()[0] == "lang":  # type: ignore[index]
            self._next()
        return Literal(value)

    def _anonymous(self) -> Bnode:
        node = self._fresh_bnode()
        if self._peek() != ("punct", "]"):
            self._predicate_object_list(node)
        self._expect("]")
        return node

    def _collection(self) -> Term:
        items: list[Term] = []
        while self._peek() != ("punct", ")"):
            items.append(self._term())
        self._expect(")")
        if not items:
            return Iri(RDF_NIL)
        head = self._fresh_bnode()
        node: Term = head
        for index, item in enumerate(items):
            self.triples.append((node, Iri(RDF_FIRST), item))
            tail: Term = Iri(RDF_NIL) if index == len(items) - 1 else self._fresh_bnode()
            self.triples.append((node, Iri(RDF_REST), tail))
            node = tail
        return head

    def _expand(self, pname: str) -> str:
        prefix, _, local = pname.partition(":")
        if prefix not in self.prefixes:
            raise TurtleError(f"undeclared prefix {prefix!r} in {pname!r}")
        return self.prefixes[prefix] + local


def parse_turtle(text: str) -> tuple[dict[str, str], list[Triple]]:
    """Read Turtle into `(prefixes, triples)`; prefixed names arrive already expanded."""
    parser = _Parser(_tokenize(text))
    parser.parse()
    return parser.prefixes, parser.triples


def collection_items(triples: list[Triple], head: Term) -> list[Term]:
    """Walk an `rdf:first`/`rdf:rest` chain back into the list it was written as."""
    first = {s: o for s, p, o in triples if p.value == RDF_FIRST}
    rest = {s: o for s, p, o in triples if p.value == RDF_REST}
    items: list[Term] = []
    node = head
    while isinstance(node, Bnode) and node in first:
        items.append(first[node])
        node = rest.get(node, Iri(RDF_NIL))
    return items


def escape_literal(value: str) -> str:
    """Escape a string for a Turtle quoted literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

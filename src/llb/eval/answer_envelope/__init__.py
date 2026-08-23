"""The declared RAG answer contract and the boundary that validates it (typed-rag-answer-envelope).

The free-text answer path recovers every answer-side signal after the fact -- a status from regex
markers, an abstention from refusal stems, citations scraped out of prose, claims re-segmented by
punctuation. This package replaces the guessing with a declaration: `models` is the contract,
`boundary` parses and validates a completion into it exactly once (spending at most one bounded
repair reprompt), `lane` turns that outcome into the graph's terminal state, and `metrics` reads
the answer-side numbers off the declared fields instead of the prose.

Nothing here checks whether the declaration is TRUE -- that is the next capability's job. This
package answers one question: did the model emit the requested shape, and what did it declare?
"""

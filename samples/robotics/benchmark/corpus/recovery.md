# Recoverable write failures

A failed idempotent assignment may be proposed again after a fresh state read and policy check. An
unacknowledged non-idempotent additive write is different: treat its outcome as unknown, reconcile,
and do not retry the same proposal.


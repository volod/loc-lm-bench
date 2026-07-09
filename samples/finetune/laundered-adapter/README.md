# laundered-adapter fixture

`adapter_manifest.json` here **claims** a clean `tuning`-only training set. The registry fixture
[`../registry/registry.jsonl`](../registry/registry.jsonl) records what this adapter was *actually*
trained on: the `final`-split item `sample-final-item`.

The contamination guard resolves adapter provenance through the registry, so it refuses this
adapter even though the manifest beside its weights was hand-edited to look clean. This is the
difference between "operator-supplied" and "recorded" digests.

See [`../poisoned-adapter/`](../poisoned-adapter/) for the simpler case where the manifest itself
declares the protected split, which the guard also refuses when the adapter is unregistered.

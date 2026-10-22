"""vectormailbox — hybrid search over your own Gmail inbox.

Dense cosine similarity over EmbeddingGemma 2 vectors fused with SQLite FTS5
BM25, over a local SQLite index built by `vectormailbox index`, served as a
mailbox-shaped UI by `vectormailbox gui`.

This init is deliberately empty apart from the version. Importing one module
must not drag in another module's dependencies: `search`, `index` and the eval
tools work on a machine that never installed a web stack, so `gui.py` is
imported by the `gui` command and nowhere near here."""

from ._version import __version__

__all__ = ["__version__"]

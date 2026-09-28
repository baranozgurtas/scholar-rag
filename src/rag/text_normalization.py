"""Unicode normalization shared by ingestion, citation tags and citation matching.

PDF text extraction keeps typographic ligatures such as "ﬀ" (U+FB00) and
"ﬁ" (U+FB01). A model that copies a tag usually writes plain "ff"/"fi", so a
title stored as "Eﬀects" must compare equal to "Effects". NFKC folds these
compatibility characters to their plain forms; it is applied to both sides
of every comparison and does not relax any matching rule.
"""

from __future__ import annotations

import unicodedata


def nfkc(text: str) -> str:
    """NFKC-normalize `text` (ligatures, full-width forms, non-breaking spaces)."""
    return unicodedata.normalize("NFKC", text)


__all__ = ["nfkc"]

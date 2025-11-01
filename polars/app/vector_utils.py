"""
Utility helpers for working with PostgreSQL pgvector columns.
"""

from typing import Sequence


def to_pgvector(vector: Sequence[float]) -> str:
    """
    Convert an iterable of floats into the textual format expected by pgvector.

    Args:
        vector: Sequence of numeric values representing a vector.

    Returns:
        String formatted as '[v1,v2,...]' suitable for insertion into pgvector columns.
    """
    # Ensure we always produce at least an empty vector literal
    if not vector:
        return "[]"

    components = ",".join(format(float(value), ".12g") for value in vector)
    return f"[{components}]"

def batch_ranges(limit: int, batch_size: int):
    """
    Yield contiguous batch index ranges.

    Args:
    -----
    limit : int
        Upper bound of index range.
    batch_size : int
        Number of elements per batch.

    Returns:
    --------
    Iterator[tuple[int, int]]
        Start and end indices for each batch, end-exclusive.
    """
    for start in range(0, limit, batch_size):
        yield start, min(start + batch_size, limit)

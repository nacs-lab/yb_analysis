"""Circular buffer for fixed-size N-D arrays.

Port of YbExptCtrl/data_processing/ArrayQueue.m
"""

import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer backed by a pre-allocated numpy array.

    Parameters
    ----------
    capacity : int
        Maximum number of elements.
    elem_shape : tuple of int
        Shape of each element (e.g. (H, W) for images).
    dtype : str or np.dtype
        Data type for storage.
    """

    def __init__(self, capacity, elem_shape, dtype='float64'):
        self.capacity = capacity
        self.elem_shape = tuple(elem_shape)
        self._buffer = np.zeros((capacity, *self.elem_shape), dtype=dtype)
        self._head = 0   # oldest element index
        self._tail = 0   # next write index
        self._count = 0

    def push(self, x):
        """Enqueue element. Overwrites oldest if full."""
        x = np.asarray(x)
        assert x.shape == self.elem_shape, (
            f"Expected shape {self.elem_shape}, got {x.shape}"
        )

        if self._count == self.capacity:
            # drop oldest
            self._head = (self._head + 1) % self.capacity
            self._count -= 1

        self._buffer[self._tail] = x
        self._tail = (self._tail + 1) % self.capacity
        self._count += 1

    def pop(self):
        """Dequeue and return oldest element."""
        if self._count == 0:
            raise IndexError("Buffer is empty")
        x = self._buffer[self._head].copy()
        self._head = (self._head + 1) % self.capacity
        self._count -= 1
        return x

    def get_last_n(self, n):
        """Return the last n elements as an (n, *elem_shape) array."""
        if n > self._count:
            raise IndexError(
                f"Only {self._count} items but asked for {n}"
            )
        start = (self._tail - n) % self.capacity
        indices = np.arange(start, start + n) % self.capacity
        return self._buffer[indices].copy()

    def size(self):
        """Current number of stored elements."""
        return self._count

    def is_empty(self):
        return self._count == 0

    def is_full(self):
        return self._count == self.capacity

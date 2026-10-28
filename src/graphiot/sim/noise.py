"""Counter-based, command-independent sensor noise.

Determinism and counterfactual replay require that the noise added to a sensor
at tick ``k`` depend only on (seed, signal, k) — never on the command history.
Then removing one command and replaying changes the physics but not the noise,
so the oracle's leave-one-out test is clean (paper Section V-B).

Each signal gets its own PCG64 stream seeded from (seed, signal). The plant
draws exactly one sample per signal per tick via :meth:`tick`, so the k-th
sample of a signal is fixed regardless of what commands were applied.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


class NoiseField:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._rng: dict[str, np.random.Generator] = {}
        self._cache: dict[str, float] = {}
        self._k = 0

    def _stream(self, name: str) -> np.random.Generator:
        rng = self._rng.get(name)
        if rng is None:
            s = (self.seed ^ _stable_hash(name)) & ((1 << 64) - 1)
            rng = np.random.Generator(np.random.PCG64(s))
            self._rng[name] = rng
        return rng

    def tick(self, names: list[str]) -> None:
        """Advance one tick: draw a fresh N(0,1) for every named signal."""
        self._cache = {n: float(self._stream(n).standard_normal()) for n in names}
        self._k += 1

    def value(self, name: str, sigma: float) -> float:
        """The current tick's noise for ``name`` scaled by ``sigma`` (0 if unseen)."""
        return self._cache.get(name, 0.0) * sigma

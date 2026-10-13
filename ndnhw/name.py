from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True, order=True)
class NDNName:
    """Immutable representation of an NDN hierarchical name.

    Internally, a name is a tuple of string components. The root name is the empty tuple.
    """

    components: Tuple[str, ...]

    @classmethod
    def parse(cls, value: str | Iterable[str] | "NDNName") -> "NDNName":
        if isinstance(value, NDNName):
            return value
        if isinstance(value, str):
            s = value.strip()
            if s in {"", "/"}:
                return cls(())
            parts = tuple(part for part in s.split("/") if part)
            return cls(parts)
        return cls(tuple(str(x) for x in value))

    @property
    def depth(self) -> int:
        return len(self.components)

    def prefix(self, depth: int) -> "NDNName":
        if depth < 0:
            raise ValueError("depth must be non-negative")
        return NDNName(self.components[:depth])

    def prefixes(self, include_root: bool = False) -> list["NDNName"]:
        start = 0 if include_root else 1
        return [self.prefix(i) for i in range(start, self.depth + 1)]

    def parent(self) -> "NDNName | None":
        if not self.components:
            return None
        return NDNName(self.components[:-1])

    def startswith(self, prefix: "NDNName") -> bool:
        prefix = NDNName.parse(prefix)
        return self.components[: prefix.depth] == prefix.components

    def __str__(self) -> str:
        if not self.components:
            return "/"
        return "/" + "/".join(self.components)

    def __repr__(self) -> str:
        return f"NDNName({str(self)!r})"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Optional

from .name import NDNName


@dataclass
class TrieNode:
    name: NDNName
    node_id: int
    parent_id: Optional[int]
    children: Dict[str, int] = field(default_factory=dict)
    is_fib_prefix: bool = False
    fib_weight: float = 0.0
    access_load: float = 0.0

    @property
    def depth(self) -> int:
        return self.name.depth


class PrefixTrie:
    """Prefix trie with node ids stable under insertion order."""

    def __init__(self) -> None:
        self.nodes: list[TrieNode] = [TrieNode(name=NDNName.parse("/"), node_id=0, parent_id=None)]
        self._name_to_id: dict[NDNName, int] = {NDNName.parse("/"): 0}

    @property
    def root_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, node_id: int) -> TrieNode:
        return self.nodes[node_id]

    def node_id(self, name: NDNName | str) -> int | None:
        return self._name_to_id.get(NDNName.parse(name))

    def insert_path(self, name: NDNName | str) -> int:
        name = NDNName.parse(name)
        current = self.root_id
        current_name = NDNName.parse("/")
        for comp in name.components:
            node = self.nodes[current]
            if comp in node.children:
                current = node.children[comp]
                current_name = self.nodes[current].name
                continue
            current_name = NDNName.parse(current_name.components + (comp,))
            new_id = len(self.nodes)
            self.nodes[current].children[comp] = new_id
            self.nodes.append(TrieNode(name=current_name, node_id=new_id, parent_id=current))
            self._name_to_id[current_name] = new_id
            current = new_id
        return current

    def add_fib_prefix(self, prefix: NDNName | str, weight: float = 0.0, access_load: float | None = None) -> int:
        node_id = self.insert_path(prefix)
        node = self.nodes[node_id]
        node.is_fib_prefix = True
        node.fib_weight += float(weight)
        if access_load is None:
            access_load = weight
        node.access_load += float(access_load)
        return node_id

    def iter_nodes(self, include_root: bool = False) -> Iterator[TrieNode]:
        start = 0 if include_root else 1
        yield from self.nodes[start:]

    def ancestors(self, node_id: int, include_self: bool = True, include_root: bool = False) -> list[int]:
        out = []
        cur: int | None = node_id if include_self else self.nodes[node_id].parent_id
        while cur is not None:
            if include_root or cur != self.root_id:
                out.append(cur)
            cur = self.nodes[cur].parent_id
        out.reverse()
        return out

    def descendants(self, node_id: int, include_self: bool = True) -> list[int]:
        out = []
        if include_self:
            out.append(node_id)
        stack = list(self.nodes[node_id].children.values())
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.nodes[cur].children.values())
        return out

    def lpm(self, name: NDNName | str) -> int | None:
        """Return node id of true longest FIB prefix matching name, if any."""
        name = NDNName.parse(name)
        current = self.root_id
        best: int | None = self.root_id if self.nodes[self.root_id].is_fib_prefix else None

        for comp in name.components:
            node = self.nodes[current]
            if comp not in node.children:
                break
            current = node.children[comp]
            if self.nodes[current].is_fib_prefix:
                best = current
        return best

    def matching_selected_prefixes(self, name: NDNName | str, selected: set[int]) -> list[int]:
        name = NDNName.parse(name)
        current = self.root_id
        matches = []
        if current in selected:
            matches.append(current)
        for comp in name.components:
            node = self.nodes[current]
            if comp not in node.children:
                break
            current = node.children[comp]
            if current in selected:
                matches.append(current)
        return matches

    def deepest_selected_match(self, name: NDNName | str, selected: set[int]) -> int | None:
        matches = self.matching_selected_prefixes(name, selected)
        return matches[-1] if matches else None

    def is_prefix_closed(self, selected: set[int], require_root: bool = False) -> bool:
        if require_root and self.root_id not in selected:
            return False
        for v in selected:
            parent = self.nodes[v].parent_id
            while parent is not None and parent != self.root_id:
                if parent not in selected:
                    return False
                parent = self.nodes[parent].parent_id
        return True

    def closure(self, selected: Iterable[int], include_root: bool = False) -> set[int]:
        closed: set[int] = set()
        for node_id in selected:
            closed.update(self.ancestors(node_id, include_self=True, include_root=include_root))
        return closed

    def max_fib_depth_below(self, node_id: int) -> int:
        return max((self.nodes[v].depth for v in self.descendants(node_id) if self.nodes[v].is_fib_prefix), default=self.nodes[node_id].depth)

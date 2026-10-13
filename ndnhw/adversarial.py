from __future__ import annotations

from .name import NDNName
from .trie import PrefixTrie
from .trace import InterestTrace


def unsafe_shortcut_case() -> tuple[InterestTrace, PrefixTrie, set[int]]:
    """Hardware has /a/b but misses deeper true FIB prefix /a/b/c."""
    trie = PrefixTrie()
    ab = trie.add_fib_prefix("/a/b", weight=10)
    trie.add_fib_prefix("/a/b/c", weight=90)
    trace = InterestTrace([NDNName.parse("/a/b/c/file")] * 90 + [NDNName.parse("/a/b/x")] * 10)

    selected = trie.closure({ab}, include_root=False)
    return trace, trie, selected


def deep_prefix_cost_case() -> tuple[InterestTrace, PrefixTrie]:
    """A deep hot prefix has hidden ancestor cost, competing with shallow prefixes."""
    trie = PrefixTrie()
    trie.add_fib_prefix("/a/b/c/d/e/f", weight=100)
    trie.add_fib_prefix("/x", weight=35)
    trie.add_fib_prefix("/y", weight=35)
    trie.add_fib_prefix("/z", weight=35)
    trace = InterestTrace(
        [NDNName.parse("/a/b/c/d/e/f/item")] * 100
        + [NDNName.parse("/x/item")] * 35
        + [NDNName.parse("/y/item")] * 35
        + [NDNName.parse("/z/item")] * 35
    )
    return trace, trie


def bank_fits_memory_fails_case() -> tuple[InterestTrace, PrefixTrie]:
    """Memory can hold hot prefixes, but bank load can make all-hot placement infeasible."""
    trie = PrefixTrie()
    trie.add_fib_prefix("/hot/a", weight=100, access_load=100)
    trie.add_fib_prefix("/hot/b", weight=100, access_load=100)
    trie.add_fib_prefix("/hot/c", weight=100, access_load=100)
    trie.add_fib_prefix("/cool/a", weight=45, access_load=45)
    trie.add_fib_prefix("/cool/b", weight=45, access_load=45)
    trace = InterestTrace(
        [NDNName.parse("/hot/a/1")] * 100
        + [NDNName.parse("/hot/b/1")] * 100
        + [NDNName.parse("/hot/c/1")] * 100
        + [NDNName.parse("/cool/a/1")] * 45
        + [NDNName.parse("/cool/b/1")] * 45
    )
    return trace, trie

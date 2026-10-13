from ndnhw.name import NDNName
from ndnhw.trie import PrefixTrie


def test_name_parse_and_prefixes():
    name = NDNName.parse("/a/b/c")
    assert name.depth == 3
    assert str(name.prefix(2)) == "/a/b"
    assert [str(p) for p in name.prefixes()] == ["/a", "/a/b", "/a/b/c"]


def test_trie_lpm():
    trie = PrefixTrie()
    ab = trie.add_fib_prefix("/a/b", weight=1)
    abc = trie.add_fib_prefix("/a/b/c", weight=1)

    assert trie.lpm("/a/b/c/d") == abc
    assert trie.lpm("/a/b/x") == ab
    assert trie.lpm("/x") is None

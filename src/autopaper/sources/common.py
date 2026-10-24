from __future__ import annotations

from autopaper.textutil import collapse_ws


def local_name(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def direct_child(element, name: str):
    if element is None:
        return None
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def direct_children(element, name: str) -> list:
    if element is None:
        return []
    return [child for child in list(element) if local_name(child.tag) == name]


def child_text(element, name: str) -> str:
    child = direct_child(element, name)
    if child is None:
        return ""
    return collapse_ws("".join(child.itertext()))

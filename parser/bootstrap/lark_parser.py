import json

from pathlib import Path
from typing import Any

from lark import Lark, Tree, Token
from lark.tree import Meta


GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
lark = Lark.open(
    str(GRAMMAR_PATH),
    parser="earley",
    ambiguity="resolve",
    start="program",
    propagate_positions=True,
)


def token_to_dict(token: Token) -> dict[str, Any]:
    return {
        "name": token.type,
        "value": token.value,
        "meta": {
            "start": {"line": token.line, "column": token.column},
            "end": {"line": token.end_line, "column": token.end_column},
        },
    }


def tree_to_dict(tree: Tree) -> dict[str, Any]:
    if hasattr(tree.meta, "line"):
        start = {"line": tree.meta.line, "column": tree.meta.column}
        end = {"line": tree.meta.end_line, "column": tree.meta.end_column}
    else:
        # IDK why this happens but it is very rare, might be a lark bug?
        # No use fixing it if we're just gonna remove lark tho
        start = {"line": 0, "column": 0}
        end = {"line": 0, "column": 0}

    item = {
        "name": tree.data.value if isinstance(tree.data, Token) else tree.data,
        "children": [],
        "meta": {
            "start": start,
            "end": end,
        },
    }

    for child in tree.children:
        if isinstance(child, Token):
            item["children"].append(token_to_dict(child))
        elif child is None:
            item["children"].append(None)
        else:
            item["children"].append(tree_to_dict(child))

    return item


def parse(path: Path) -> str:
    tree = lark.parse(path.read_text())
    return json.dumps(tree_to_dict(tree))

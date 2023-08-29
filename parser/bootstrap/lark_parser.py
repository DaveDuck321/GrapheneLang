import json
from pathlib import Path
from typing import Any

from lark import Lark, Token, Tree, UnexpectedInput
from lark.tree import Meta

from codegen.user_facing_errors import InvalidSyntax

GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
lark_instance = Lark.open(
    str(GRAMMAR_PATH),
    parser="earley",
    ambiguity="resolve",
    start="program",
    propagate_positions=True,
)


def meta_to_dict(meta: Meta | Token) -> dict[str, Any]:
    if not hasattr(meta, "line"):
        # If a tree has no tokens the meta is empty, default to 0,0
        return meta_to_dict(Token("", ""))

    return {
        "start": {"line": meta.line, "column": meta.column},
        "end": {"line": meta.end_line, "column": meta.end_column},
    }


def token_to_dict(token: Token) -> dict[str, Any]:
    return {"name": token.type, "value": token.value, "meta": meta_to_dict(token)}


def tree_to_dict(tree: Tree) -> dict[str, Any]:
    children = []
    for child in tree.children:
        if isinstance(child, Token):
            children.append(token_to_dict(child))
        elif isinstance(child, Tree):
            children.append(tree_to_dict(child))
        else:
            children.append(None)

    return {
        "name": tree.data.value if isinstance(tree.data, Token) else tree.data,
        "children": children,
        "meta": meta_to_dict(tree.meta),
    }


def parse_and_wrap_errors(lark: Lark, path: Path) -> Tree:
    with open(path, encoding="utf-8") as source_file:
        file_content = source_file.read()

    try:
        return lark.parse(file_content)
    except UnexpectedInput as exc:
        assert isinstance(exc.pos_in_stream, int)
        error_pos = exc.pos_in_stream

        this_line_start_pos = file_content[:error_pos].rfind("\n")
        this_line_end_pos = error_pos + file_content[error_pos:].find("\n")

        error_message_context = [
            file_content[this_line_start_pos + 1 : this_line_end_pos]
        ]

        # Is there only white space on this line before the error message?
        if file_content[this_line_start_pos:error_pos].isspace():
            # Then we should also print the previous line (where the error probably occurred)
            previous_line = " "
            previous_line_end = error_pos - 1
            while previous_line.isspace():
                line_end = file_content[:previous_line_end].rfind("\n")
                if line_end == -1:
                    break

                previous_line = file_content[line_end + 1 : previous_line_end]
                previous_line_end = line_end

            if not previous_line.isspace():
                error_message_context.insert(0, previous_line)

        caret_pos = error_pos - this_line_start_pos - 1
        error_message_context.append(caret_pos * " " + "^")

        raise InvalidSyntax(error_message_context, exc.line) from exc


def parse(path: Path) -> str:
    tree = parse_and_wrap_errors(lark_instance, path)
    return json.dumps(tree_to_dict(tree))

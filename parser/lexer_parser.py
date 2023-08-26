import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

parse_fn: Callable[[Path], str] | None = None


@dataclass
class JsonConvertible:
    @classmethod
    def is_convertible(cls, json_dict: dict) -> bool:
        for field in fields(cls):
            if field.name not in json_dict:
                return False
        return True


@dataclass
class FilePosition(JsonConvertible):
    line: int
    column: int


@dataclass
class Meta(JsonConvertible):
    start: FilePosition
    end: FilePosition


@dataclass
class Token(JsonConvertible):
    # TODO: rename to token
    name: str
    value: str
    meta: Meta


@dataclass
class Tree(JsonConvertible):
    name: str
    children: list[Any]
    meta: Meta


class Transformer:
    def __init__(self, visit_tokens=False) -> None:
        self._visit_tokens = visit_tokens

    def transform(self, tree: Tree):
        my_children = []
        for child in tree.children:
            if child is None:
                my_children.append(None)
            elif isinstance(child, Token):
                if self._visit_tokens and hasattr(self, child.name):
                    my_children.append(getattr(self, child.name)(child))
                else:
                    my_children.append(child)
            else:
                assert isinstance(child, Tree)
                my_children.append(self.transform(child))

        new_tree = Tree(tree.name, my_children, tree.meta)
        if not hasattr(self, tree.name):
            return new_tree

        return getattr(self, tree.name)(new_tree)


class Interpreter:
    def __init__(self, visit_tokens=False) -> None:
        self._visit_tokens = visit_tokens

    def visit(self, tree: Tree | Token):
        if not hasattr(self, tree.name):
            if isinstance(tree, Token):
                return tree

            return self.visit_children(tree)

        return getattr(self, tree.name)(tree)

    def visit_children(self, tree: Tree):
        for child in tree.children:
            if child is not None:
                self.visit(child)


def json_to_tree(json_data: str) -> Tree:
    def decoder(obj: dict):
        for type_ in (Tree, Token, Meta, FilePosition):
            if type_.is_convertible(obj):
                return type_(**obj)

        raise NotImplementedError()

    return json.loads(json_data, object_hook=decoder)


def init_lexer_parser(self_hosted: bool):
    global parse_fn

    if self_hosted:
        raise NotImplementedError()
    else:
        from .bootstrap.lark_parser import parse

    parse_fn = parse


def run_lexer_parser(path: Path) -> Tree:
    assert parse_fn is not None, "The lexer parser is uninitialized"
    return json_to_tree(parse_fn(path))

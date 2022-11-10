from dataclasses import dataclass

from lark import Lark, Tree, Token
from lark.visitors import Interpreter


@dataclass
class FunctionDefinition:
    args: list
    return_type_name: str


def extract_leaf_value(tree: Tree) -> str:
    assert len(tree.children) == 1
    assert isinstance(tree.children[0], Token)

    return str(tree.children[0])


def get_unique_child(tree: Tree, name: str) -> Tree:
    matches = [child for child in tree.children if child.data == name]

    if not matches:
        raise RuntimeError(f"Tree {tree} doesn't have child {name}")
    if len(matches) > 1:
        raise RuntimeError(f"Tree {tree} has more than one children {name}")

    return matches[0]


class SymbolTableGenerator(Interpreter):
    def __init__(self) -> None:
        super().__init__()

        self.functions: dict[FunctionDefinition] = {}

    def named_function(self, tree: Tree):
        fn_name_tree = get_unique_child(tree, "function_name")
        fn_name = extract_leaf_value(fn_name_tree)

        # TODO parse this.
        fn_args = get_unique_child(tree, "function_arguments").children

        # TODO parse more complex types.
        fn_type_tree = get_unique_child(tree, "type")
        type_name_tree = get_unique_child(fn_type_tree, "type_name")
        fn_type = extract_leaf_value(type_name_tree)

        # TODO add support for overloading.
        if fn_name in self.functions:
            raise RuntimeError(f"Duplicate function name {fn_name}")

        self.functions[fn_name] = FunctionDefinition(fn_args, fn_type)


l = Lark.open("grammar.lark", parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())

    symbol_table_gen = SymbolTableGenerator()
    symbol_table_gen.visit(tree)

    function_table = symbol_table_gen.functions

    print(function_table)

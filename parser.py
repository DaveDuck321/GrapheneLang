import codegen

from lark import Lark, Tree, Token
from lark.visitors import Interpreter


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


def extract_named_leaf_value(tree: Tree, name: str) -> str:
    return extract_leaf_value(get_unique_child(tree, name))


class SymbolTableGenerator(Interpreter):
    def __init__(self) -> None:
        super().__init__()

        self._program = codegen.Program()

    def named_function(self, tree: Tree):
        fn_name = extract_named_leaf_value(tree, "function_name")

        fn_args = []
        fn_arg_trees = get_unique_child(tree, "function_arguments").children
        for arg_name, arg_type in zip(fn_arg_trees[::2], fn_arg_trees[1::2]):
            # TODO parse adhoc/ generic types.
            name = extract_leaf_value(arg_name)
            type_name = extract_named_leaf_value(arg_type, "type_name")

            type = self._program.lookup_type(type_name)
            fn_args.append(codegen.Variable(name, type))

        # TODO parse adhoc/ generic types.
        fn_return_type_tree = get_unique_child(tree, "function_return_type")
        fn_return_type_name = extract_named_leaf_value(fn_return_type_tree, "type_name")
        fn_return_type = self._program.lookup_type(fn_return_type_name)

        # Build the function
        function_obj = codegen.Function(fn_name, fn_args, fn_return_type)
        self._program.add_function(function_obj)


l = Lark.open("grammar.lark", parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())

    symbol_table_gen = SymbolTableGenerator()
    symbol_table_gen.visit(tree)

    function_table = symbol_table_gen.functions

    # TODO: now codegen function subexpressions

    print(function_table)

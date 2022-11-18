from lark import Lark, Token, Tree
from lark.visitors import Interpreter

import codegen


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
    # TODO: also parse typedefs
    def __init__(self, program: codegen.Program) -> None:
        super().__init__()

        self._program = program
        self._function_body_trees: list[tuple[codegen.Function, Tree]] = []

    def get_function_body_trees(self):
        return self._function_body_trees

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

        fn_signature = codegen.FunctionSignature(fn_name, fn_args)

        # TODO parse adhoc/ generic types.
        fn_return_type_tree = get_unique_child(tree, "function_return_type")
        fn_return_type_name = extract_named_leaf_value(fn_return_type_tree, "type_name")
        fn_return_type = self._program.lookup_type(fn_return_type_name)

        # Build the function
        function_obj = codegen.Function(fn_signature, fn_return_type)
        self._program.add_function(function_obj)

        # Save the body to parse later (TODO: maybe forward declarations should be possible?)
        fn_body_tree = get_unique_child(tree, "scope")
        self._function_body_trees.append((function_obj, fn_body_tree))


l = Lark.open("grammar.lark", parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())

    program = codegen.Program()
    symbol_table_gen = SymbolTableGenerator(program)
    symbol_table_gen.visit(tree)

    unparsed_function_bodies = symbol_table_gen.get_function_body_trees()
    print(unparsed_function_bodies)

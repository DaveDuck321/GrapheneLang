from itertools import count

from lark import Lark, Token, Tree
from lark.visitors import Interpreter, Transformer_InPlace

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


class ScopeTransformer(Transformer_InPlace):
    def __init__(self, program: codegen.Program) -> None:
        super().__init__(True)

        self.program = program

        self.expr_id_iter = count()
        self.expressions: list[codegen.Expression] = []

    def SIGNED_INT(self, value: str) -> codegen.ConstantExpression:
        const_expr = codegen.ConstantExpression(
            next(self.expr_id_iter), codegen.IntType(), int(value)
        )
        self.expressions.append(const_expr)

        return const_expr

    def return_statement(
        self, sub_expressions: list[codegen.Expression]
    ) -> codegen.ReturnExpression:
        assert len(sub_expressions) <= 1
        returned_expr = sub_expressions[0] if sub_expressions else None

        ret_expr = codegen.ReturnExpression(next(self.expr_id_iter), returned_expr)
        self.expressions.append(ret_expr)

        return ret_expr

    def function_call(self, children: list[Tree]) -> codegen.FunctionCallExpression:
        fn_name = extract_leaf_value(children[0])
        args = children[1].children

        args_for_sig = []
        for arg in args:
            assert isinstance(arg, codegen.TypedExpression)
            # FIXME name?
            var = codegen.Variable("", arg.type)
            args_for_sig.append(var)

        fn_sig = codegen.FunctionSignature(fn_name, args_for_sig)
        fn = self.program.lookup_function(fn_sig)

        call_expr = codegen.FunctionCallExpression(next(self.expr_id_iter), fn, args)
        self.expressions.append(call_expr)

        return call_expr

    def ESCAPED_STRING(self, string: str) -> codegen.StringConstant:
        assert string[0] == '"' and string[-1] == '"'
        identifier = self.program.add_string(string[1:-1])

        str_const = codegen.StringConstant(next(self.expr_id_iter), identifier)
        self.expressions.append(str_const)

        return str_const


l = Lark.open("grammar.lark", parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())

    program = codegen.Program()
    symbol_table_gen = SymbolTableGenerator(program)
    symbol_table_gen.visit(tree)

    program.add_function(
        codegen.Function(
            codegen.FunctionSignature(
                "puts", [codegen.Variable("string", program.lookup_type("string"))]
            ),
            program.lookup_type("int"),
        )
    )

    for function, body in symbol_table_gen.get_function_body_trees():
        et = ScopeTransformer(program)
        et.transform(body)

        print(body.pretty())

        function.expressions = et.expressions

    with open("demo.ll", "w") as file:
        file.writelines(program.generate_ir())

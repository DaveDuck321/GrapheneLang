from dataclasses import dataclass

from lark import Lark, Token, Tree
from lark.visitors import Interpreter, Transformer_InPlace

import codegen as cg


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
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._program = program
        self._function_body_trees: list[tuple[cg.Function, Tree]] = []

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
            fn_args.append(cg.Variable(name, type))

        fn_signature = cg.FunctionSignature(fn_name, fn_args)

        # TODO parse adhoc/ generic types.
        fn_return_type_tree = get_unique_child(tree, "function_return_type")
        fn_return_type_name = extract_named_leaf_value(fn_return_type_tree, "type_name")
        fn_return_type = self._program.lookup_type(fn_return_type_name)

        # Build the function
        function_obj = cg.Function(fn_signature, fn_return_type)
        self._program.add_function(function_obj)

        # Save the body to parse later (TODO: maybe forward declarations should be possible?)
        fn_body_tree = get_unique_child(tree, "scope")
        self._function_body_trees.append((function_obj, fn_body_tree))


@dataclass
class FlattenedExpression:
    subexpressions: list[cg.TypedExpression]

    def add_parent(self, expression: cg.TypedExpression) -> "FlattenedExpression":
        self.subexpressions.append(expression)
        return self

    def expression(self) -> cg.TypedExpression:
        return self.subexpressions[-1]

    def type(self) -> cg.TypedExpression:
        return self.expression().type


class ExpressionTransformer(Transformer_InPlace):
    def __init__(self, program: cg.Program, function: cg.Function) -> None:
        super().__init__(visit_tokens=True)

        self._program = program
        self._function = function

    def SIGNED_INT(self, value: str) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(
            self._function.get_next_expr_id(), cg.IntType(), int(value)
        )
        return FlattenedExpression([const_expr])

    def function_call(self, children: list[Tree]) -> FlattenedExpression:
        fn_name = extract_leaf_value(children[0])

        flattened_expr = FlattenedExpression([])
        fn_args = []
        args_for_signature = []
        for arg in children[1].children:
            assert isinstance(arg, FlattenedExpression)
            # FIXME name?
            var = cg.Variable("", arg.type())
            args_for_signature.append(var)
            fn_args.append(arg.expression())

            flattened_expr.subexpressions.extend(arg.subexpressions)

        fn_signature = cg.FunctionSignature(fn_name, args_for_signature)
        fn = self._program.lookup_function(fn_signature)

        call_expr = cg.FunctionCallExpression(
            self._function.get_next_expr_id(), fn, fn_args
        )

        return flattened_expr.add_parent(call_expr)

    def ESCAPED_STRING(self, string: str) -> FlattenedExpression:
        assert string[0] == '"' and string[-1] == '"'
        identifier = self._program.add_string(string[1:-1])

        str_const = cg.StringConstant(self._function.get_next_expr_id(), identifier)

        return FlattenedExpression([str_const])


def generate_standalone_expression(
    program: cg.Program, function: cg.Function, body: Tree
) -> None:
    assert len(body.children) == 1
    ExpressionTransformer(program, function).transform(body)

    flattened_expr = body.children[0]
    assert isinstance(flattened_expr, FlattenedExpression)

    for subexpression in flattened_expr.subexpressions:
        function.expressions.append(subexpression)


def generate_return_statement(
    program: cg.Program, function: cg.Function, body: Tree
) -> None:
    if not body.children:
        expr = cg.ReturnExpression(function.get_next_expr_id())
        function.expressions.append(expr)
        return

    assert len(body.children) == 1
    ExpressionTransformer(program, function).transform(body)

    flattened_expr = body.children[0]
    assert isinstance(flattened_expr, FlattenedExpression)

    for subexpression in flattened_expr.subexpressions:
        function.expressions.append(subexpression)

    expr = cg.ReturnExpression(function.get_next_expr_id(), flattened_expr.expression())

    function.expressions.append(expr)


def generate_function_body(program: cg.Program, function: cg.Function, body: Tree):
    assert body.data == "scope"

    generators = {
        "return_statement": generate_return_statement,
        "expression": generate_standalone_expression,
    }
    for line in body.children:
        generators[line.data](program, function, line)


l = Lark.open("grammar.lark", parser="lalr", start="program")

with open("demo.c3") as source:
    tree = l.parse(source.read())

    print(tree.pretty())

    program = cg.Program()
    symbol_table_gen = SymbolTableGenerator(program)
    symbol_table_gen.visit(tree)

    program.add_function(
        cg.Function(
            cg.FunctionSignature(
                "puts", [cg.Variable("string", program.lookup_type("string"))]
            ),
            program.lookup_type("int"),
        )
    )

    for function, body in symbol_table_gen.get_function_body_trees():
        generate_function_body(program, function, body)

    with open("demo.ll", "w") as file:
        file.writelines(program.generate_ir())

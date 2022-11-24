from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

from lark import Lark, Token, Tree
from lark.visitors import Interpreter, Transformer_InPlace, v_args
from lark.exceptions import VisitError

import codegen as cg
from errors import GrapheneError


class ErrorWithLineInfo(ValueError):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


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

    @v_args(inline=True)
    def named_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        fn = self._build_function(name_tree, args_tree, return_type_tree, False)
        self._program.add_function(fn)

        # Save the body to parse later (TODO: maybe forward declarations should be possible?)
        self._function_body_trees.append((fn, body_tree))

    @v_args(inline=True)
    def foreign_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
    ) -> None:
        fn = self._build_function(name_tree, args_tree, return_type_tree, True)
        self._program.add_function(fn)

    def _build_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        foreign: bool,
    ) -> None:
        fn_name = extract_leaf_value(name_tree)

        fn_args = []
        fn_arg_trees = args_tree.children
        for arg_name, arg_type in zip(fn_arg_trees[::2], fn_arg_trees[1::2]):
            # TODO parse adhoc/ generic types.
            name = extract_leaf_value(arg_name)
            type_name = extract_named_leaf_value(arg_type, "type_name")

            type = self._program.lookup_type(type_name)
            fn_args.append(cg.Variable(name, type))

        # TODO parse adhoc/ generic types.
        fn_return_type_name = extract_named_leaf_value(return_type_tree, "type_name")
        fn_return_type = self._program.lookup_type(fn_return_type_name)

        # Build the function
        return cg.Function(fn_name, fn_args, fn_return_type, foreign)


@dataclass
class FlattenedExpression:
    subexpressions: list[cg.TypedExpression]

    def add_parent(self, expression: cg.TypedExpression) -> "FlattenedExpression":
        self.subexpressions.append(expression)
        return self

    def expression(self) -> cg.TypedExpression:
        return self.subexpressions[-1]

    def type(self) -> cg.Type:
        return self.expression().type


class ExpressionTransformer(Transformer_InPlace):
    def __init__(
        self, program: cg.Program, function: cg.Function, scope: cg.Scope
    ) -> None:
        super().__init__(visit_tokens=True)

        self._program = program
        self._function = function
        self._scope = scope

    def SIGNED_INT(self, value: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(
            self._function.get_next_expr_id(), cg.IntType(), int(value)
        )
        return FlattenedExpression([const_expr])

    @v_args(inline=True)
    def bool_constant(self, value: str) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(
            self._function.get_next_expr_id(), cg.BoolType(), value == "true"
        )
        return FlattenedExpression([const_expr])

    @v_args(inline=True)
    def function_call(self, name_tree: Tree, args_tree: Tree) -> FlattenedExpression:
        fn_name = extract_leaf_value(name_tree)

        flattened_expr = FlattenedExpression([])
        arg_types_for_lookup = []
        fn_call_args = []
        for arg in args_tree.children:
            assert isinstance(arg, FlattenedExpression)
            arg_types_for_lookup.append(arg.type())
            fn_call_args.append(arg.expression())

            flattened_expr.subexpressions.extend(arg.subexpressions)

        function = self._program.lookup_function(fn_name, arg_types_for_lookup)
        call_expr = cg.FunctionCallExpression(
            self._function.get_next_expr_id(), function, fn_call_args
        )

        return flattened_expr.add_parent(call_expr)

    def ESCAPED_STRING(self, string: Token) -> FlattenedExpression:
        assert string[0] == '"' and string[-1] == '"'
        identifier = self._program.add_string(string[1:-1])

        str_const = cg.StringConstant(self._function.get_next_expr_id(), identifier)

        return FlattenedExpression([str_const])

    @v_args(inline=True)
    def accessed_variable_name(self, var_name: Token) -> FlattenedExpression:
        var = self._scope.search_for_variable(var_name)

        var_access = cg.VariableAccess(self._function.get_next_expr_id(), var)

        return FlattenedExpression([var_access])


def generate_standalone_expression(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:
    assert len(body.children) == 1
    ExpressionTransformer(program, function, scope).transform(body)

    flattened_expr = body.children[0]
    assert isinstance(flattened_expr, FlattenedExpression)

    scope.add_generatable(flattened_expr.subexpressions)


def generate_return_statement(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:
    if not body.children:
        expr = cg.ReturnStatement(function.get_next_expr_id())
        scope.add_generatable(expr)
        return

    assert len(body.children) == 1
    ExpressionTransformer(program, function, scope).transform(body)

    flattened_expr = body.children[0]
    assert isinstance(flattened_expr, FlattenedExpression)
    scope.add_generatable(flattened_expr.subexpressions)

    expr = cg.ReturnStatement(function.get_next_expr_id(), flattened_expr.expression())
    scope.add_generatable(expr)


def generate_variable_declaration(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:
    def parse_variable_declaration(
        name_tree: Tree,
        type_tree: Tree,
        value: Optional[FlattenedExpression] = None,  # FIXME lark placeholders.
    ) -> FlattenedExpression:
        # Extract variable name.
        var_name = extract_leaf_value(name_tree)
        type_name = extract_named_leaf_value(type_tree, "type_name")

        # Extract variable type. TODO add support for non-type_name types.
        var_type = program.lookup_type(type_name)

        var = cg.StackVariable(var_name, var_type, False, value is not None)
        scope.add_variable(var)

        if value is None:
            return FlattenedExpression([])

        # Initialize variable.
        assignment_expr = cg.VariableAssignment(
            function.get_next_expr_id(), var, value.expression()
        )
        # TODO: FlattenedExpression no-longer represents an expression here
        return value.add_parent(assignment_expr)

    # Need to parse value first.
    ExpressionTransformer(program, function, scope).transform(body)

    flattened_expr = parse_variable_declaration(*body.children)
    scope.add_generatable(flattened_expr.subexpressions)


def generate_if_statement(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:

    condition_tree, scope_tree = body.children
    ExpressionTransformer(program, function, scope).transform(condition_tree)

    assert len(condition_tree.children) == 1
    condition_expr = condition_tree.children[0]

    scope.add_generatable(condition_expr.subexpressions)

    inner_scope = cg.Scope(function.get_next_expr_id(), scope)
    generate_body(program, function, inner_scope, scope_tree)

    if_statement = cg.IfStatement(
        function.get_next_expr_id(), condition_expr.expression(), inner_scope
    )
    scope.add_generatable(if_statement)


def generate_scope_body(
    program: cg.Program, function: cg.Function, outer_scope: cg.Scope, body: Tree
) -> None:
    inner_scope = cg.Scope(function.get_next_expr_id(), outer_scope)
    generate_body(program, function, inner_scope, body)
    outer_scope.add_generatable(inner_scope)


def generate_body(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:

    generators = {
        "return_statement": generate_return_statement,
        "expression": generate_standalone_expression,
        "scope": generate_scope_body,
        "variable_declaration": generate_variable_declaration,
        "if_statement": generate_if_statement,
    }

    for line in body.children:
        try:
            generators[line.data](program, function, scope, line)
        except VisitError as e:
            if not isinstance(e.orig_exc, GrapheneError):
                raise e from e.orig_exc
            raise ErrorWithLineInfo(e.orig_exc.message, line.meta.line) from None

        except GrapheneError as e:
            raise ErrorWithLineInfo(e.message, line.meta.line) from None


def generate_function_body(program: cg.Program, function: cg.Function, body: Tree):
    generate_body(program, function, function.top_level_scope, body)


def generate_ir_from_source(file_path: Path):
    grammar_path = Path(__file__).parent / "grammar.lark"
    lark = Lark.open(
        grammar_path, parser="lalr", start="program", propagate_positions=True
    )
    tree = lark.parse(file_path.open().read())

    print(tree.pretty())

    program = cg.Program()
    symbol_table_gen = SymbolTableGenerator(program)
    symbol_table_gen.visit(tree)

    try:
        for function, body in symbol_table_gen.get_function_body_trees():
            generate_function_body(program, function, body)
    except ErrorWithLineInfo as e:
        print(
            f'File "{file_path.absolute()}", line {e.line}, in "{function.get_signature()}"',
            file=sys.stderr,
        )
        print(f"   {e.message}", file=sys.stderr)
        exit(1)

    return "\n".join(program.generate_ir())


if __name__ == "__main__":
    with open("demo.c3") as source:
        ir = generate_ir_from_source(source.read())
        open("demo.ll", "w").write(ir)

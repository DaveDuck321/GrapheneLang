import sys
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

from lark import Lark, Token, Tree
from lark.exceptions import VisitError
from lark.visitors import Interpreter, Transformer, Transformer_InPlace, v_args

import codegen as cg
from codegen.user_facing_errors import (
    ErrorWithLineInfo,
    FailedLookupError,
    GenericArgumentCountError,
    GenericHasGenericAnnotation,
    GrapheneError,
    InvalidInitializerListLength,
    RepeatedGenericName,
    assert_else_throw,
)


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


def in_pairs(iterable: Iterable) -> Iterable:
    # [iter(...), iter(...)] would make two different list_iterator objects.
    # We only want one.
    chunks = [iter(iterable)] * 2

    return zip(*chunks, strict=True)


class TypeTransformer(Transformer):
    def __init__(
        self, program: cg.Program, generic_mapping: dict[str, cg.Type]
    ) -> None:
        super().__init__(visit_tokens=False)

        self._program = program
        self._generic_mapping = generic_mapping

    @v_args(inline=True)
    def type(self, child: cg.Type) -> cg.Type:
        assert isinstance(child, cg.Type)
        return child

    @v_args(inline=True)
    def type_name(self, name: Token, type_map: Optional[Tree]) -> cg.Type:
        assert isinstance(name, Token)

        if name in self._generic_mapping:
            assert_else_throw(type_map is None, GenericHasGenericAnnotation(name))
            return self._generic_mapping[name]

        generic_args: list[cg.Type] = [] if type_map is None else type_map.children  # type: ignore
        return self._program.lookup_type(name, generic_args)

    @v_args(inline=True)
    def ref_type(self, value_type: cg.Type) -> cg.Type:
        assert isinstance(value_type, cg.Type)

        return value_type.to_reference()

    @v_args(inline=True)
    def stack_array_type(
        self, element_type: cg.Type, *dimension_tokens: Token
    ) -> cg.Type:
        dimensions: list[int] = [int(dimension.value) for dimension in dimension_tokens]
        return cg.Type(cg.ArrayDefinition(element_type, dimensions))

    @v_args(inline=True)
    def heap_array_type(
        self, element_type: cg.Type, *dimension_tokens: Token
    ) -> cg.Type:
        dimensions: list[int] = [cg.ArrayDefinition.UNKNOWN_DIMENSION]
        for dimension in dimension_tokens:
            dimensions.append(int(dimension.value))

        # A heap array must always be passed by reference
        underlying_array = cg.Type(cg.ArrayDefinition(element_type, dimensions))
        return underlying_array.to_reference()

    def struct_type(self, member_trees: list[Tree]) -> cg.Type:
        members: list[cg.Parameter] = []
        for member_name_tree, member_type in in_pairs(member_trees):
            assert isinstance(member_type, cg.Type)

            member_name = extract_leaf_value(member_name_tree)
            assert isinstance(member_name, str)

            members.append(cg.Parameter(member_name, member_type))

        return cg.Type(cg.StructDefinition(members))

    @classmethod
    def parse(
        cls,
        program: cg.Program,
        tree: Tree,
        type_map: Optional[dict[str, cg.Type]] = None,
    ) -> cg.Type:
        if type_map is None:
            type_map = {}

        try:
            result = cls(program, type_map).transform(tree)
        except VisitError as exc:
            raise exc.orig_exc

        assert isinstance(result, cg.Type)

        return result


class ParseTypeDefinitions(Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._program = program

    def _typedef(self, type_name: str, generics: list[Token], rhs_tree: Tree) -> None:
        available_generics: list[str] = []
        for generic_name in generics:
            assert isinstance(generic_name, Token)

            assert_else_throw(
                generic_name.value not in available_generics,
                RepeatedGenericName(generic_name, type_name),
            )

            available_generics.append(generic_name.value)

        def type_parser(name_prefix: str, concrete_types: list[cg.Type]) -> cg.Type:
            assert_else_throw(
                len(concrete_types) == len(available_generics),
                GenericArgumentCountError(
                    type_name, len(concrete_types), len(available_generics)
                ),
            )

            mapping = dict(zip(available_generics, concrete_types))
            rhs = TypeTransformer.parse(self._program, rhs_tree, mapping)
            return rhs.new_from_typedef(name_prefix, concrete_types)

        self._program.add_type(type_name, type_parser)

    @v_args(inline=True)
    def typedef(
        self, type_name: Token, generic_tree: Optional[Tree], rhs_tree: Tree
    ) -> None:
        generics = [] if generic_tree is None else generic_tree.children
        return self._typedef(type_name.value, generics, rhs_tree)  # type: ignore


class ParseFunctionSignatures(Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._program = program
        self._function_body_trees: list[tuple[cg.Function, Tree]] = []

    def get_function_body_trees(self) -> list[tuple[cg.Function, Tree]]:
        return self._function_body_trees

    @v_args(inline=True)
    def named_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        self._parse_function(name_tree, args_tree, return_type_tree, body_tree, False)

    @v_args(inline=True)
    def operator_function(
        self, op_tree: Tree, args_tree: Tree, return_type_tree: Tree, body_tree: Tree
    ) -> None:
        self._parse_function(op_tree, args_tree, return_type_tree, body_tree, False)

    @v_args(inline=True)
    def foreign_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
    ) -> None:
        self._parse_function(name_tree, args_tree, return_type_tree, None, True)

    def _parse_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Optional[Tree],
        foreign: bool,
    ) -> None:
        try:
            func = self._build_function(name_tree, args_tree, return_type_tree, foreign)
            self._program.add_function(func)

            # Save the body to parse later (TODO: maybe forward declarations
            # should be possible?)
            if body_tree is not None:
                self._function_body_trees.append((func, body_tree))
        except GrapheneError as exc:
            # Not ideal but better than nothing.
            raise ErrorWithLineInfo(
                exc.message, name_tree.meta.line, extract_leaf_value(name_tree)
            ) from exc

    def _build_function(
        self,
        name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        foreign: bool,
    ) -> cg.Function:
        fn_name = extract_leaf_value(name_tree)

        fn_args: list[cg.Parameter] = []
        fn_arg_trees = args_tree.children
        for arg_name_tree, arg_type_tree in in_pairs(fn_arg_trees):
            arg_name = extract_leaf_value(arg_name_tree)
            arg_type = TypeTransformer.parse(self._program, arg_type_tree)

            fn_args.append(cg.Parameter(arg_name, arg_type))

        fn_return_type = TypeTransformer.parse(self._program, return_type_tree)

        # Build the function
        return cg.Function(fn_name, fn_args, fn_return_type, foreign)


@dataclass
class FlattenedExpression:
    subexpressions: list[cg.Generatable]

    def add_parent(self, expression: cg.TypedExpression) -> "FlattenedExpression":
        self.subexpressions.append(expression)
        return self

    def expression(self) -> cg.TypedExpression:
        assert isinstance(self.subexpressions[-1], cg.TypedExpression)
        return self.subexpressions[-1]

    def type(self) -> cg.Type:
        return self.expression().type


@dataclass
class InitializerList:
    exprs: list[FlattenedExpression]
    names: Optional[list[str]]

    def __len__(self) -> int:
        if self.names:
            assert len(self.exprs) == len(self.names)

        return len(self.exprs)


class ExpressionTransformer(Transformer_InPlace):
    def __init__(
        self, program: cg.Program, function: cg.Function, scope: cg.Scope
    ) -> None:
        super().__init__(visit_tokens=True)

        self._program = program
        self._function = function
        self._scope = scope

    @v_args(inline=True)
    def expression(
        self, value: FlattenedExpression | InitializerList
    ) -> FlattenedExpression | InitializerList:
        assert isinstance(value, (FlattenedExpression, InitializerList))
        return value

    def SIGNED_INT(self, value: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.IntType(), value)
        return FlattenedExpression([const_expr])

    @v_args(inline=True)
    def bool_constant(self, value: str) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.BoolType(), value)
        return FlattenedExpression([const_expr])

    @v_args(inline=True)
    def operator_use(
        self, lhs: FlattenedExpression, operator: Token, rhs: FlattenedExpression
    ) -> FlattenedExpression:
        assert isinstance(lhs, FlattenedExpression)
        assert isinstance(rhs, FlattenedExpression)

        flattened_expr = FlattenedExpression([])
        flattened_expr.subexpressions.extend(lhs.subexpressions)
        flattened_expr.subexpressions.extend(rhs.subexpressions)

        call_expr = self._program.lookup_call_expression(
            operator.value,
            [lhs.expression(), rhs.expression()],
        )
        return flattened_expr.add_parent(call_expr)

    @v_args(inline=True)
    def unary_operator_use(
        self, operator: Token, rhs: FlattenedExpression
    ) -> FlattenedExpression:
        assert isinstance(rhs, FlattenedExpression)

        flattened_expr = FlattenedExpression(rhs.subexpressions)

        call_expr = self._program.lookup_call_expression(
            operator.value,
            [rhs.expression()],
        )
        return flattened_expr.add_parent(call_expr)

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

        call_expr = self._program.lookup_call_expression(fn_name, fn_call_args)
        return flattened_expr.add_parent(call_expr)

    def ESCAPED_STRING(self, string: Token) -> FlattenedExpression:
        assert string[0] == '"' and string[-1] == '"'
        identifier = self._program.add_string(string[1:-1])

        str_const = cg.ConstantExpression(cg.StringType(), identifier)
        return FlattenedExpression([str_const])

    @v_args(inline=True)
    def accessed_variable_name(self, var_name: Token) -> FlattenedExpression:
        var = self._scope.search_for_variable(var_name)

        assert_else_throw(var is not None, FailedLookupError("variable", var_name))
        assert var is not None  # Make the type checker happy.

        var_ref = cg.VariableReference(var)
        return FlattenedExpression([var_ref])

    def ensure_pointer_is_available(self, expr: FlattenedExpression):
        # Copy expression to stack if it is not a pointer
        if expr.type().is_pointer:
            return expr

        temp_var = cg.StackVariable("", expr.type(), True, True)
        self._scope.add_variable(temp_var)

        expr.subexpressions.append(cg.VariableAssignment(temp_var, expr.expression()))
        return expr.add_parent(cg.VariableReference(temp_var))

    @v_args(inline=True)
    def array_index_access(
        self, lhs: FlattenedExpression, *index_exprs: FlattenedExpression
    ) -> FlattenedExpression:

        lhs = self.ensure_pointer_is_available(lhs)
        result = FlattenedExpression([*lhs.subexpressions])

        cg_indices: list[cg.TypedExpression] = []
        for index_expr in index_exprs:
            cg_indices.append(index_expr.expression())
            result.subexpressions.extend(index_expr.subexpressions)

        return result.add_parent(cg.ArrayIndexAccess(lhs.expression(), cg_indices))

    @v_args(inline=True)
    def struct_member_access(
        self, lhs: FlattenedExpression, member_tree: Tree
    ) -> FlattenedExpression:
        member_name = extract_leaf_value(member_tree)

        struct_access = cg.StructMemberAccess(lhs.expression(), member_name)
        return lhs.add_parent(struct_access)

    @v_args(inline=True)
    def borrow_operator_use(self, lhs: FlattenedExpression):
        borrow = cg.Borrow(lhs.expression())
        return lhs.add_parent(borrow)

    def struct_initializer_without_names(
        self, objects: list[FlattenedExpression]
    ) -> InitializerList:
        assert isinstance(objects, list)

        return InitializerList(objects, None)

    def struct_initializer_with_names(
        self, objects: list[FlattenedExpression | Token]
    ) -> InitializerList:
        assert isinstance(objects, list)

        # Use zip to transpose a list of pairs into a pair of lists.
        names, exprs = list(map(list, zip(*in_pairs(objects))))

        return InitializerList(exprs, names)

    @v_args(inline=True)
    def adhoc_struct_initialization(
        self, init_list: InitializerList
    ) -> InitializerList:
        assert isinstance(init_list, InitializerList)

        # TODO handle empty initializer lists.
        if not init_list.exprs:
            raise NotImplementedError()

        return init_list


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
        # FIXME change IntType() to void once we implement that.
        expr = cg.ReturnStatement(cg.IntType())
        scope.add_generatable(expr)
        return

    assert len(body.children) == 1
    ExpressionTransformer(program, function, scope).transform(body)

    flattened_expr = body.children[0]
    assert isinstance(flattened_expr, FlattenedExpression)
    scope.add_generatable(flattened_expr.subexpressions)

    expr = cg.ReturnStatement(
        function.get_signature().return_type, flattened_expr.expression()
    )
    scope.add_generatable(expr)


def generate_variable_declaration(
    is_const: bool,
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
) -> None:
    def parse_variable_declaration(
        name_tree: Tree,
        type_tree: Tree,
        rhs: Optional[
            FlattenedExpression | InitializerList
        ] = None,  # FIXME lark placeholders.
    ) -> None:
        # Extract variable name and type.
        var_name = extract_leaf_value(name_tree)
        var_type = TypeTransformer.parse(program, type_tree)

        var = cg.StackVariable(var_name, var_type, is_const, rhs is not None)
        scope.add_variable(var)

        # Initialize variable.
        if isinstance(rhs, FlattenedExpression):
            scope.add_generatable(rhs.subexpressions)
            scope.add_generatable(cg.VariableAssignment(var, rhs.expression()))

        # Initialize struct.
        elif isinstance(rhs, InitializerList):
            # TODO proper errors.
            assert isinstance(var_type.definition, cg.StructDefinition)
            assert_else_throw(
                var_type.definition.member_count == len(rhs),
                InvalidInitializerListLength(
                    len(rhs), var_type.definition.member_count
                ),
            )

            def assign_to_member(expr: FlattenedExpression, member_name: str) -> None:
                scope.add_generatable(expr.subexpressions)

                var_ref = cg.VariableReference(var)
                scope.add_generatable(var_ref)

                struct_access = cg.StructMemberAccess(var_ref, member_name)
                scope.add_generatable(struct_access)

                var_assignment = cg.Assignment(struct_access, expr.expression())
                scope.add_generatable(var_assignment)

            if rhs.names:
                for name, expr in zip(rhs.names, rhs.exprs):
                    assign_to_member(expr, name)
            else:
                for idx, expr in enumerate(rhs.exprs):
                    member = var_type.definition.get_member_by_index(idx)
                    assign_to_member(expr, member.name)

    name_tree, type_tree, value_tree = body.children
    expression_value = (
        ExpressionTransformer(program, function, scope).transform(value_tree)
        if value_tree is not None
        else None
    )
    parse_variable_declaration(name_tree, type_tree, expression_value)


def generate_if_statement(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:
    condition_tree, scope_tree = body.children
    ExpressionTransformer(program, function, scope).transform(condition_tree)

    assert len(condition_tree.children) == 1
    condition_expr = condition_tree.children[0]
    assert isinstance(condition_expr, FlattenedExpression)

    scope.add_generatable(condition_expr.subexpressions)

    inner_scope = cg.Scope(function.get_next_scope_id(), scope)
    generate_body(program, function, inner_scope, scope_tree)

    if_statement = cg.IfStatement(condition_expr.expression(), inner_scope)
    scope.add_generatable(if_statement)


def generate_assignment(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:

    lhs_tree, rhs_tree = body.children
    lhs = ExpressionTransformer(program, function, scope).transform(lhs_tree)
    rhs = ExpressionTransformer(program, function, scope).transform(rhs_tree)
    assert isinstance(lhs, FlattenedExpression)
    assert isinstance(rhs, FlattenedExpression)

    scope.add_generatable(lhs.subexpressions)
    scope.add_generatable(rhs.subexpressions)
    scope.add_generatable(cg.Assignment(lhs.expression(), rhs.expression()))


def generate_scope_body(
    program: cg.Program, function: cg.Function, outer_scope: cg.Scope, body: Tree
) -> None:
    inner_scope = cg.Scope(function.get_next_scope_id(), outer_scope)
    generate_body(program, function, inner_scope, body)
    outer_scope.add_generatable(inner_scope)


def generate_body(
    program: cg.Program, function: cg.Function, scope: cg.Scope, body: Tree
) -> None:

    generators = {
        "assignment": generate_assignment,
        "const_declaration": partial(generate_variable_declaration, True),
        "expression": generate_standalone_expression,
        "if_statement": generate_if_statement,
        "return_statement": generate_return_statement,
        "scope": generate_scope_body,
        "variable_declaration": partial(generate_variable_declaration, False),
    }

    for line in body.children:
        try:
            generators[line.data](program, function, scope, line)
        except VisitError as exc:
            if isinstance(exc.orig_exc, GrapheneError):
                raise ErrorWithLineInfo(
                    exc.orig_exc.message,
                    line.meta.line,
                    function.get_signature().user_facing_name,
                ) from exc.orig_exc
            raise exc.orig_exc
        except GrapheneError as exc:
            raise ErrorWithLineInfo(
                exc.message,
                line.meta.line,
                function.get_signature().user_facing_name,
            ) from exc


def generate_function_body(program: cg.Program, function: cg.Function, body: Tree):
    generate_body(program, function, function.top_level_scope, body)


def generate_ir_from_source(file_path: Path, debug_compiler: bool = False) -> str:
    grammar_path = Path(__file__).parent / "grammar.lark"
    lark = Lark.open(
        str(grammar_path), parser="lalr", start="program", propagate_positions=True
    )
    tree = lark.parse(file_path.open().read())

    print(tree.pretty())

    program = cg.Program()

    try:
        # TODO: these stages can be combined if we require forward declaration
        # FIXME: allow recursive types
        ParseTypeDefinitions(program).visit(tree)
        fn_pass = ParseFunctionSignatures(program)
        fn_pass.visit(tree)

        for function, body in fn_pass.get_function_body_trees():
            generate_function_body(program, function, body)
    except ErrorWithLineInfo as exc:
        if debug_compiler:
            traceback.print_exc()
            print("~~~ User-facing error message ~~~")

        print(
            f"File '{file_path.absolute()}', line {exc.line}, in '{exc.context}'",
            file=sys.stderr,
        )
        print(f"   {exc.message}", file=sys.stderr)
        sys.exit(1)

    return "\n".join(program.generate_ir())

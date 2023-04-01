import sys
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional, TypeGuard

from lark import Lark, Token, Tree
from lark.exceptions import VisitError
from lark.visitors import Interpreter, Transformer, v_args

import codegen as cg
from codegen.user_facing_errors import (
    CannotAssignToInitializerList,
    CircularImportException,
    DoubleReferenceError,
    ErrorWithLineInfo,
    FailedLookupError,
    FileDoesNotExistException,
    FileIsAmbiguousException,
    GenericArgumentCountError,
    GenericHasGenericAnnotation,
    GrapheneError,
    InitializerListTypeDeductionFailure,
    InvalidInitializerListAssignment,
    InvalidInitializerListLength,
    InvalidMainReturnType,
    MissingFunctionReturn,
    RepeatedGenericName,
    SubstitutionFailure,
    VoidVariableDeclaration,
)


class ResolvedPath(str):
    def __new__(cls, path: Path) -> "ResolvedPath":
        return str.__new__(cls, str(path.resolve()))


def in_pairs(iterable: Iterable) -> Iterable:
    # [iter(...), iter(...)] would make two different list_iterator objects.
    # We only want one.
    chunks = [iter(iterable)] * 2

    return zip(*chunks, strict=True)


def inline_and_wrap_user_facing_errors(context: str):
    def wrapper(func, _, children, meta):
        try:
            func(*children)
        except GrapheneError as exc:
            raise ErrorWithLineInfo(
                exc.message,
                meta.line,
                context,
            ) from exc

    return v_args(wrapper=wrapper)


def search_include_path_for_file(
    relative_file_path: str, include_path: list[Path]
) -> Optional[ResolvedPath]:
    matching_files: set[ResolvedPath] = set()
    for path in include_path:
        file_path = path / relative_file_path
        if file_path.exists():
            matching_files.add(ResolvedPath(file_path))

    if len(matching_files) == 0:
        return None

    if len(matching_files) != 1:
        raise FileIsAmbiguousException(relative_file_path, matching_files)

    return matching_files.pop()


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
            if type_map is not None:
                raise GenericHasGenericAnnotation(name)

            return self._generic_mapping[name]

        generic_args: list[cg.Type] = [] if type_map is None else type_map.children  # type: ignore
        return self._program.lookup_type(name, generic_args)

    @v_args(inline=True)
    def ref_type(self, value_type: cg.Type) -> cg.Type:
        assert isinstance(value_type, cg.Type)

        if value_type.is_borrowed_reference:
            raise DoubleReferenceError(value_type.get_user_facing_name(True))

        return value_type.take_reference()

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
        return underlying_array.take_reference()

    def struct_type(self, member_trees: list[Token | cg.Type]) -> cg.Type:
        members = []

        # Yes, this is how you are supposed to annotate unpacking products...
        m_name: str
        m_type: cg.Type
        for m_name, m_type in in_pairs(member_trees):
            if m_type.is_void:
                raise VoidVariableDeclaration(
                    "struct member", m_name, m_type.get_user_facing_name(True)
                )
            members.append(cg.Parameter(m_name, m_type))

        return cg.Type(cg.StructDefinition(members))

    @classmethod
    def parse(
        cls,
        program: cg.Program,
        tree: Tree,
        type_map: dict[str, cg.Type],
    ) -> cg.Type:
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

            if generic_name.value in available_generics:
                raise RepeatedGenericName(generic_name, type_name)

            available_generics.append(generic_name.value)

        def type_parser(name_prefix: str, concrete_types: list[cg.Type]) -> cg.Type:
            if len(concrete_types) != len(available_generics):
                raise GenericArgumentCountError(
                    type_name, len(concrete_types), len(available_generics)
                )

            mapping = {
                str(generic): concrete_type
                for generic, concrete_type in zip(generics, concrete_types)
            }

            rhs = TypeTransformer.parse(self._program, rhs_tree, mapping)
            return rhs.new_from_typedef(name_prefix, concrete_types)

        self._program.add_type(type_name, type_parser)

    @inline_and_wrap_user_facing_errors("typedef")
    def generic_typedef(
        self, generic_tree: Optional[Tree], type_name: Token, rhs_tree: Tree
    ) -> None:
        generics = [] if generic_tree is None else generic_tree.children
        return self._typedef(type_name.value, generics, rhs_tree)  # type: ignore

    @inline_and_wrap_user_facing_errors("typedef<...>")
    def specialized_typedef(
        self, type_name: Token, specialization_tree: Tree, rhs_tree: Tree
    ) -> None:
        specialization = []
        for specialization_type_tree in specialization_tree.children:
            # Note: partial specializations are not allowed, generic mapping is empty
            specialization.append(
                TypeTransformer.parse(self._program, specialization_type_tree, {})
            )

        def type_parser(name_prefix: str, concrete_types: list[cg.Type]) -> cg.Type:
            assert concrete_types == specialization
            rhs = TypeTransformer.parse(self._program, rhs_tree, {})
            return rhs.new_from_typedef(name_prefix, concrete_types)

        self._program.add_specialized_type(type_name, type_parser, specialization)


class ParseFunctionSignatures(Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._program = program
        self._function_body_trees: list[
            tuple[cg.Function, Tree, dict[str, cg.Type]]
        ] = []

    def get_function_body_trees(
        self,
    ) -> list[tuple[cg.Function, Tree, dict[str, cg.Type]]]:
        return self._function_body_trees

    @inline_and_wrap_user_facing_errors("function[...] signature")
    def generic_named_function(
        self,
        generic_names_tree: Tree,
        generic_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        assert isinstance(generic_name, str)
        generic_names: list[str] = generic_names_tree.children  # type: ignore
        self._parse_generic_function_impl(
            generic_name, generic_names, args_tree, return_type_tree, body_tree
        )

    @inline_and_wrap_user_facing_errors("@operator[...] signature")
    def generic_operator_function(
        self,
        generic_names_tree: Tree,
        op_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        assert isinstance(op_name, str)
        generic_names: list[str] = generic_names_tree.children  # type: ignore
        self._parse_generic_function_impl(
            op_name, generic_names, args_tree, return_type_tree, body_tree
        )

    def _parse_generic_function_impl(
        self,
        generic_name: str,
        generic_names: list[str],
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        def try_parse_fn_from_specialization(
            fn_name: str, concrete_specializations: list[cg.Type]
        ) -> Optional[cg.Function]:
            assert generic_name == fn_name
            if len(generic_names) != len(concrete_specializations):
                return None

            generic_mapping = dict(zip(generic_names, concrete_specializations))
            try:
                function = self._build_function(
                    fn_name, args_tree, return_type_tree, False, generic_mapping
                )
            except SubstitutionFailure:
                return None  # SFINAE

            generate_function_body(self._program, function, body_tree, generic_mapping)
            return function

        def try_deduce_specialization(
            fn_name: str, arguments: list[cg.Type]
        ) -> Optional[list[cg.Type]]:
            assert generic_name == fn_name
            deduced_mapping: dict[str, cg.Type] = {}

            for provided_arg_type, (_, arg_type_tree) in zip(
                arguments, in_pairs(args_tree.children)
            ):
                arg_type_in_generic_definition, _ = arg_type_tree.children[0].children

                # TODO: type pattern matching here
                #       Atm this is a simple string compare
                assert isinstance(arg_type_in_generic_definition, Token)
                if arg_type_in_generic_definition not in generic_names:
                    # This argument is not a generic
                    continue

                # The argument is a generic, deduce it's type

                # Have we already deduced a different type?
                if arg_type_in_generic_definition in deduced_mapping:
                    if (
                        deduced_mapping[arg_type_in_generic_definition]
                        != provided_arg_type
                    ):
                        return None  # SFINAE

                deduced_mapping[arg_type_in_generic_definition] = provided_arg_type

            # Convert the deduced mapping into a specialization
            deduced_specialization: list[cg.Type] = []
            for generic in generic_names:
                deduced_specialization.append(deduced_mapping[generic])

            return deduced_specialization

        self._program.add_generic_function(
            generic_name,
            cg.GenericFunctionParser(
                generic_name,
                try_deduce_specialization,
                try_parse_fn_from_specialization,
            ),
        )

    @inline_and_wrap_user_facing_errors("function signature")
    def specialized_named_function(
        self,
        function_name_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        name, specialization_tree = function_name_tree.children
        if specialization_tree is not None:
            raise NotImplementedError()

        assert isinstance(name, str)
        self._parse_function(name, args_tree, return_type_tree, body_tree, False)

    @inline_and_wrap_user_facing_errors("@operator signature")
    def specialized_operator_function(
        self,
        op_name: Token,
        specialization: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        if specialization is not None:
            raise NotImplementedError()

        self._parse_function(op_name, args_tree, return_type_tree, body_tree, False)

    @inline_and_wrap_user_facing_errors("foreign signature")
    def foreign_function(
        self,
        fn_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
    ) -> None:
        self._parse_function(fn_name, args_tree, return_type_tree, None, True)

    def _parse_function(
        self,
        fn_name: str,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Optional[Tree],
        foreign: bool,
    ) -> None:
        func = self._build_function(fn_name, args_tree, return_type_tree, foreign)
        self._program.add_function(func)

        # Save the body to parse later (TODO: maybe forward declarations
        # should be possible?)
        if body_tree is not None:
            self._function_body_trees.append((func, body_tree, {}))

    def _build_function(
        self,
        fn_name: str,
        args_tree: Tree,
        return_type_tree: Tree,
        foreign: bool,
        generic_mapping: dict[str, cg.Type] = {},
    ) -> cg.Function:
        fn_args: list[cg.Parameter] = []
        fn_arg_trees = args_tree.children
        for arg_name, arg_type_tree in in_pairs(fn_arg_trees):
            assert isinstance(arg_name, Token)
            arg_type = TypeTransformer.parse(
                self._program, arg_type_tree, generic_mapping
            )

            if arg_type.is_void:
                raise VoidVariableDeclaration(
                    "argument", arg_name, arg_type.get_user_facing_name(True)
                )

            fn_args.append(cg.Parameter(arg_name, arg_type))

        fn_return_type = TypeTransformer.parse(
            self._program, return_type_tree, generic_mapping
        )

        # Build the function
        fn_obj = cg.Function(
            fn_name, fn_args, fn_return_type, foreign, list(generic_mapping.values())
        )

        # main() must always return an int
        if (
            fn_obj.get_signature().is_main()
            and fn_obj.get_signature().return_type != cg.IntType()
        ):
            raise InvalidMainReturnType(
                fn_obj.get_signature().return_type.get_user_facing_name(True)
            )

        return fn_obj


class ParseImports(Interpreter):
    def __init__(
        self,
        lark: Lark,
        program: cg.Program,
        include_path: list[Path],
        included_from: list[ResolvedPath],
        already_processed: set[ResolvedPath],
    ) -> None:
        super().__init__()

        self._lark = lark
        self._program = program
        self._include_path = include_path
        self._included_from = included_from
        self._already_processed = already_processed

    def require_once(self, path_tree: Tree) -> None:
        path_token = path_tree.children[0]
        assert isinstance(path_token, Token)

        try:
            self._require_once_impl(path_token[1:-1])
        except GrapheneError as exc:
            raise ErrorWithLineInfo(
                exc.message, path_tree.meta.line, f"@require_once {path_token}"
            )

    def _require_once_impl(self, path_str: str) -> None:
        file_path = search_include_path_for_file(path_str, self._include_path)

        if file_path is None:
            raise FileDoesNotExistException(path_str)

        if file_path in self._included_from:
            index = self._included_from.index(file_path)
            if index == 0:
                file_with_conflicting_import = "<compile target>"
            else:
                file_with_conflicting_import = self._included_from[index - 1]

            raise CircularImportException(file_path, file_with_conflicting_import)

        if file_path in self._already_processed:
            # File is already in translation unit, nothing to do
            return

        append_file_to_program(
            self._lark,
            self._program,
            file_path,
            self._include_path[:-1],  # Last element is always '.'
            self._included_from,
            self._already_processed,
        )


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
        return self.expression().underlying_type


def is_flattened_expression_iterable(
    exprs: Iterable[Any],
) -> TypeGuard[Iterable[FlattenedExpression]]:
    # https://github.com/python/mypy/issues/3497#issuecomment-1083747764
    return all(isinstance(expr, FlattenedExpression) for expr in exprs)


@dataclass
class InitializerList:
    exprs: list[FlattenedExpression]
    names: Optional[list[str]]

    def __len__(self) -> int:
        if self.names:
            assert len(self.exprs) == len(self.names)

        return len(self.exprs)

    @property
    def user_facing_name(self) -> str:
        type_names = [expr.type().get_user_facing_name(False) for expr in self.exprs]

        members = (
            [f"{name}: {type_name}" for name, type_name in zip(self.names, type_names)]
            if self.names is not None
            else type_names
        )

        return "{" + str.join(", ", members) + "}"


class ExpressionTransformer(Transformer):
    def __init__(
        self,
        program: cg.Program,
        function: cg.Function,
        scope: cg.Scope,
        generic_mapping: dict[str, cg.Type],
    ) -> None:
        super().__init__(visit_tokens=True)

        self._program = program
        self._function = function
        self._scope = scope
        self._generic_mapping = generic_mapping

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
    def BOOL_CONSTANT(self, value: Token) -> FlattenedExpression:
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
            [],  # Don't specialize operators
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
            [],  # Don't specialize operators
            [rhs.expression()],
        )
        return flattened_expr.add_parent(call_expr)

    def _function_call_impl(
        self,
        fn_name: str,
        specialization_tree: Optional[Tree],
        fn_args: Iterable[FlattenedExpression],
    ) -> FlattenedExpression:
        flattened_expr = FlattenedExpression([])
        arg_types_for_lookup = []
        fn_call_args = []

        for arg in fn_args:
            arg_types_for_lookup.append(arg.type())
            fn_call_args.append(arg.expression())
            flattened_expr.subexpressions.extend(arg.subexpressions)

        specialization: list[cg.Type] = []
        if specialization_tree is not None:
            for concrete_type_tree in specialization_tree.children:
                specialization.append(
                    TypeTransformer.parse(
                        self._program, concrete_type_tree, self._generic_mapping
                    )
                )

        call_expr = self._program.lookup_call_expression(
            fn_name, specialization, fn_call_args
        )
        return flattened_expr.add_parent(call_expr)

    @v_args(inline=True)
    def function_call(
        self, name_tree: Tree, *args: FlattenedExpression
    ) -> FlattenedExpression:
        fn_name, specialization_tree = name_tree.children
        assert isinstance(fn_name, Token)
        assert is_flattened_expression_iterable(args)

        return self._function_call_impl(fn_name, specialization_tree, args)

    @v_args(inline=True)
    def ufcs_call(
        self, this: FlattenedExpression, name_tree: Tree, *args: FlattenedExpression
    ) -> FlattenedExpression:
        # TODO perhaps we shouldn't always borrow this, although this is a bit
        # tricky as we haven't done overload resolution yet (which depends on
        # whether we borrow or not). A solution would be to borrow if we can,
        # otherwise pass an unborrowed/const-reference and let overload
        # resolution figure it out, although this isn't very explicit.
        assert isinstance(this, FlattenedExpression)
        borrowed_this = this.add_parent(cg.BorrowExpression(this.expression()))

        fn_name, specialization_tree = name_tree.children
        assert isinstance(fn_name, str)

        fn_args = (borrowed_this, *args)
        assert is_flattened_expression_iterable(fn_args)

        return self._function_call_impl(fn_name, specialization_tree, fn_args)

    def ESCAPED_STRING(self, string: Token) -> FlattenedExpression:
        assert string[0] == '"' and string[-1] == '"'
        identifier = self._program.add_string(string[1:-1])

        str_const = cg.ConstantExpression(cg.StringType(), identifier)
        return FlattenedExpression([str_const])

    @v_args(inline=True)
    def accessed_variable_name(self, var_name: Token) -> FlattenedExpression:
        var = self._scope.search_for_variable(var_name)

        if var is None:
            raise FailedLookupError("variable", var_name)

        var_ref = cg.VariableReference(var)
        return FlattenedExpression([var_ref])

    def ensure_pointer_is_available(self, expr: FlattenedExpression):
        # Copy expression to stack if it is not a pointer
        if expr.expression().has_address:
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
        lhs_expr = lhs.expression()

        cg_indices: list[cg.TypedExpression] = []
        for index_expr in index_exprs:
            cg_indices.append(index_expr.expression())
            lhs.subexpressions.extend(index_expr.subexpressions)

        return lhs.add_parent(cg.ArrayIndexAccess(lhs_expr, cg_indices))

    @v_args(inline=True)
    def struct_member_access(
        self, lhs: FlattenedExpression, member_name: Token
    ) -> FlattenedExpression:
        assert isinstance(member_name, Token)

        struct_access = cg.StructMemberAccess(lhs.expression(), member_name)
        return lhs.add_parent(struct_access)

    @v_args(inline=True)
    def borrow_operator_use(self, lhs: FlattenedExpression):
        borrow = cg.BorrowExpression(lhs.expression())
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

        return init_list

    @staticmethod
    def parse(
        program: cg.Program,
        function: cg.Function,
        scope: cg.Scope,
        generic_mapping: dict[str, cg.Type],
        body: Tree,
    ) -> FlattenedExpression | InitializerList:
        result = ExpressionTransformer(
            program, function, scope, generic_mapping
        ).transform(body)
        assert isinstance(result, (FlattenedExpression, InitializerList))
        return result


def generate_standalone_expression(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:
    assert len(body.children) == 1
    flattened_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, body
    )
    if isinstance(flattened_expr, InitializerList):
        raise InitializerListTypeDeductionFailure()

    scope.add_generatable(flattened_expr.subexpressions)


def generate_return_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:
    if not body.children:
        expr = cg.ReturnStatement(cg.VoidType())
        scope.add_generatable(expr)
        return

    (expr,) = body.children
    flattened_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, expr
    )
    if isinstance(flattened_expr, InitializerList):
        # TODO: allow initializer lists in return statements
        raise NotImplementedError()

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
    generic_mapping: dict[str, cg.Type],
) -> None:
    var_name, type_tree, rhs_tree = body.children
    rhs: Optional[FlattenedExpression | InitializerList] = (
        ExpressionTransformer.parse(program, function, scope, generic_mapping, rhs_tree)
        if rhs_tree is not None
        else None
    )

    assert isinstance(var_name, Token)
    assert isinstance(type_tree, Tree)

    var_type = TypeTransformer.parse(program, type_tree, generic_mapping)
    if var_type.is_void:
        raise VoidVariableDeclaration(
            "variable", var_name, var_type.get_user_facing_name(True)
        )

    var = cg.StackVariable(var_name, var_type, is_const, rhs is not None)
    scope.add_variable(var)

    # Initialize variable.
    if isinstance(rhs, FlattenedExpression):
        scope.add_generatable(rhs.subexpressions)
        scope.add_generatable(cg.VariableAssignment(var, rhs.expression()))

    # Initialize struct.
    elif isinstance(rhs, InitializerList):
        if not isinstance(var_type.definition, cg.StructDefinition):
            raise InvalidInitializerListAssignment(
                var_type.get_user_facing_name(False), rhs.user_facing_name
            )

        if var_type.definition.member_count != len(rhs):
            raise InvalidInitializerListLength(
                len(rhs), var_type.definition.member_count
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

    # Unreachable if rhs has a value.
    else:
        assert rhs is None


def generate_if_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:
    condition_tree, if_scope_tree, else_scope_tree = body.children
    assert len(condition_tree.children) == 1

    condition_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, condition_tree
    )
    if isinstance(condition_expr, InitializerList):
        raise InitializerListTypeDeductionFailure()

    scope.add_generatable(condition_expr.subexpressions)

    if_scope = cg.Scope(function.get_next_scope_id(), scope)
    generate_body(program, function, if_scope, if_scope_tree, generic_mapping)

    # Note: this looks like a redundant scope when the else branch is empty but I've
    #       chosen to explicitly codegen it here so we can generate destructors in
    #       the else branch (eg. if it was moved in the if)
    else_scope = cg.Scope(function.get_next_scope_id(), scope)
    if else_scope_tree is not None:
        generate_body(program, function, else_scope, else_scope_tree, generic_mapping)

    if_statement = cg.IfElseStatement(condition_expr.expression(), if_scope, else_scope)
    scope.add_generatable(if_statement)


def generate_while_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:
    condition_tree, inner_scope_tree = body.children
    assert len(condition_tree.children) == 1

    condition_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, condition_tree
    )
    if isinstance(condition_expr, InitializerList):
        raise InitializerListTypeDeductionFailure()

    while_scope_id = function.get_next_scope_id()

    inner_scope = cg.Scope(function.get_next_scope_id(), scope)
    generate_body(program, function, inner_scope, inner_scope_tree, generic_mapping)

    scope.add_generatable(
        cg.WhileStatement(
            while_scope_id,
            condition_expr.expression(),
            condition_expr.subexpressions,
            inner_scope,
        )
    )


def generate_assignment(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:

    lhs_tree, rhs_tree = body.children
    lhs = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, lhs_tree
    )
    rhs = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, rhs_tree
    )

    if isinstance(lhs, InitializerList):
        raise CannotAssignToInitializerList()
    if isinstance(rhs, InitializerList):
        # TODO: allow initializer list assignment to structs
        raise NotImplementedError()

    scope.add_generatable(lhs.subexpressions)
    scope.add_generatable(rhs.subexpressions)
    scope.add_generatable(cg.Assignment(lhs.expression(), rhs.expression()))


def generate_scope_body(
    program: cg.Program,
    function: cg.Function,
    outer_scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:
    inner_scope = cg.Scope(function.get_next_scope_id(), outer_scope)
    generate_body(program, function, inner_scope, body, generic_mapping)
    outer_scope.add_generatable(inner_scope)


def generate_body(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
) -> None:

    generators = {
        "assignment": generate_assignment,
        "const_declaration": partial(generate_variable_declaration, True),
        "expression": generate_standalone_expression,
        "if_statement": generate_if_statement,
        "return_statement": generate_return_statement,
        "scope": generate_scope_body,
        "variable_declaration": partial(generate_variable_declaration, False),
        "while_statement": generate_while_statement,
    }

    for line in body.children:
        try:
            generators[line.data](program, function, scope, line, generic_mapping)
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


def generate_function_body(
    program: cg.Program,
    function: cg.Function,
    body: Tree,
    generic_mapping: dict[str, cg.Type],
):
    generate_body(program, function, function.top_level_scope, body, generic_mapping)

    # We cannot omit the "ret" instruction from LLVM IR. If the function returns
    # void, then we can add it ourselves, otherwise the user needs to fix it.

    if not function.top_level_scope.is_return_guaranteed():
        if function.get_signature().return_type.is_void:
            function.top_level_scope.add_generatable(cg.ReturnStatement(cg.VoidType()))
        else:
            raise MissingFunctionReturn(
                function.get_signature().user_facing_name,
                body.meta.end_line,
            )


def append_file_to_program(
    lark: Lark,
    program: cg.Program,
    file_path: ResolvedPath,
    include_path: list[Path],
    included_from: list[ResolvedPath],
    already_processed: set[ResolvedPath],
    debug_compiler: bool = False,
) -> None:
    with open(file_path, encoding="utf-8") as source_file:
        tree = lark.parse(source_file.read())

    already_processed.add(file_path)
    try:
        ParseImports(
            lark,
            program,
            include_path + [Path(file_path).parent],
            included_from + [file_path],
            already_processed,
        ).visit(tree)
        # TODO: these stages can be combined if we require forward declaration
        # FIXME: allow recursive types
        ParseTypeDefinitions(program).visit(tree)
        fn_pass = ParseFunctionSignatures(program)
        fn_pass.visit(tree)

        for function, body, generic_mapping in fn_pass.get_function_body_trees():
            generate_function_body(program, function, body, generic_mapping)

    except ErrorWithLineInfo as exc:
        if debug_compiler:
            traceback.print_exc()
            print("~~~ User-facing error message ~~~")

        print(
            f"File '{file_path}', line {exc.line}, in '{exc.context}'",
            file=sys.stderr,
        )
        print(f"    {exc.message}", file=sys.stderr)

        if included_from:
            print(file=sys.stderr)

        for file in reversed(included_from):
            print(f"Included from file '{file}'", file=sys.stderr)

        sys.exit(1)


def generate_ir_from_source(
    file_path: Path, include_path: list[Path], debug_compiler: bool = False
) -> str:
    grammar_path = Path(__file__).parent / "grammar.lark"
    lark = Lark.open(
        str(grammar_path), parser="lalr", start="program", propagate_positions=True
    )

    program = cg.Program()
    append_file_to_program(
        lark, program, ResolvedPath(file_path), include_path, [], set(), debug_compiler
    )

    return "\n".join(program.generate_ir())

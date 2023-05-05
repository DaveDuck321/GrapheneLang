import sys
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional, TypeGuard

from lark import Lark, Token, Tree
from lark.exceptions import UnexpectedInput, VisitError
from lark.visitors import Interpreter, Transformer, v_args

import codegen as cg
from codegen.user_facing_errors import (
    CircularImportException,
    DoubleReferenceError,
    ErrorWithLineInfo,
    FailedLookupError,
    FileDoesNotExistException,
    FileIsAmbiguousException,
    GenericArgumentCountError,
    GenericHasGenericAnnotation,
    GrapheneError,
    InvalidMainReturnType,
    InvalidSyntax,
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


def get_optional_children(tree: Optional[Tree]) -> list[Any]:
    if tree is None:
        return []
    return tree.children


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


def parse_generic_definition(definition: Any) -> cg.GenericArgument:
    assert isinstance(definition, Token)
    assert definition.type == "GENERIC_IDENTIFIER"
    return cg.GenericArgument(definition.value, definition.value[0] == "@")


class TypeTransformer(Transformer):
    def __init__(self, generic_args: cg.UnresolvedGenericMapping) -> None:
        super().__init__(visit_tokens=True)
        self._generic_args = generic_args

    @v_args(inline=True)
    def compile_time_constant(
        self, constant: cg.CompileTimeConstant
    ) -> cg.CompileTimeConstant:
        assert isinstance(constant, cg.CompileTimeConstant)
        return constant

    @v_args(inline=True)
    def SIGNED_INT(self, token: Token) -> cg.CompileTimeConstant:
        return cg.NumericLiteralConstant(int(token))

    @v_args(inline=True)
    def NUMERIC_IDENTIFIER(self, token: Token) -> cg.CompileTimeConstant:
        if not token.value in self._generic_args:
            assert False  # TODO: user facing error

        result = self._generic_args[token.value]
        assert isinstance(result, cg.CompileTimeConstant)  # TODO: user facing error
        return result

    @v_args(inline=True)
    def type(self, child: cg.UnresolvedType) -> cg.UnresolvedType:
        assert isinstance(child, cg.UnresolvedType)
        return child

    @v_args(inline=True)
    def type_name(
        self, name: Token, specialization_tree: Optional[Tree]
    ) -> cg.UnresolvedType:
        assert isinstance(name, Token)

        if name.value in self._generic_args:
            if specialization_tree is not None:
                raise GenericHasGenericAnnotation(name.value)

            result = self._generic_args[name.value]
            assert isinstance(result, cg.UnresolvedType)
            return result

        specialization = get_optional_children(specialization_tree)
        assert is_unresolved_specialization_list(specialization)
        return cg.UnresolvedNamedType(name.value, specialization)

    @v_args(inline=True)
    def ref_type(self, value_type: cg.UnresolvedType) -> cg.UnresolvedType:
        return cg.UnresolvedReferenceType(value_type)

    @v_args(inline=True)
    def stack_array_type(
        self, element_type: cg.UnresolvedType, *dimensions: cg.CompileTimeConstant
    ) -> cg.UnresolvedType:
        return cg.UnresolvedStackArrayType(element_type, list(dimensions))

    @v_args(inline=True)
    def heap_array_type(
        self, element_type: cg.UnresolvedType, *known_dimensions: cg.CompileTimeConstant
    ) -> cg.UnresolvedType:
        return cg.UnresolvedHeapArrayType(element_type, list(known_dimensions))

    def struct_type(
        self, member_trees: list[Token | cg.UnresolvedType]
    ) -> cg.UnresolvedType:
        members: list[tuple[str, cg.UnresolvedType]] = []

        for member_name, member_type in in_pairs(member_trees):
            assert isinstance(member_name, Token)
            assert isinstance(member_type, cg.UnresolvedType)
            members.append((member_name.value, member_type))

        return cg.UnresolvedStructType(members)

    @classmethod
    def parse_and_resolve(
        cls,
        program: cg.Program,
        tree: Tree,
        generic_mapping: cg.GenericMapping,
    ) -> cg.Type:
        unresolved_mapping: cg.UnresolvedGenericMapping = {}
        for name, item in generic_mapping.items():
            if isinstance(item, cg.Type):
                unresolved_mapping[name] = cg.UnresolvedTypeWrapper(item)
            else:
                assert isinstance(item, int)
                unresolved_mapping[name] = cg.NumericLiteralConstant(item)

        try:
            result = cls(unresolved_mapping).transform(tree)
        except VisitError as exc:
            raise exc.orig_exc

        assert isinstance(result, cg.UnresolvedType)
        return program.resolve_type(result)

    @classmethod
    def parse(
        cls, tree: Tree, generic_mapping: cg.UnresolvedGenericMapping
    ) -> cg.UnresolvedType:
        try:
            result = cls(generic_mapping).transform(tree)
        except VisitError as exc:
            raise exc.orig_exc

        assert isinstance(result, cg.UnresolvedType)
        return result

    @classmethod
    def parse_specialization(cls, tree: Tree) -> cg.UnresolvedSpecializationItem:
        try:
            result = cls({}).transform(tree)
        except VisitError as exc:
            raise exc.orig_exc

        assert isinstance(result, (cg.UnresolvedType, cg.UnresolvedSpecializationItem))
        return result


class ParseTypeDefinitions(Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._program = program

    def _generic_typedef(
        self, prefix: str, generic_definitions_tree: Tree, rhs_tree: Tree
    ) -> None:

        generic_mapping: cg.UnresolvedGenericMapping = {}
        generic_definitions = []

        for generic_tree in generic_definitions_tree.children:
            assert isinstance(generic_tree, Token)
            definition = parse_generic_definition(generic_tree)
            generic_definitions.append(definition)

            if definition.name in generic_mapping:
                raise RepeatedGenericName(generic_tree.value, prefix)

            generic_mapping[definition.name] = (
                cg.GenericValueReference(definition.name)
                if definition.is_value_arg
                else cg.UnresolvedGenericType(definition.name)
            )

        rhs_type = TypeTransformer.parse(rhs_tree, generic_mapping)
        self._program.add_generic_type_alias(prefix, generic_definitions, rhs_type)

    def _specialized_typedef(
        self,
        prefix: str,
        specialization: list[cg.UnresolvedSpecializationItem],
        rhs_tree: Tree,
    ) -> None:
        rhs_type = TypeTransformer.parse(rhs_tree, {})
        self._program.add_type_alias(prefix, specialization, rhs_type)

    @inline_and_wrap_user_facing_errors("typedef")
    def generic_typedef(
        self, generic_definitions_tree: Optional[Tree], type_name: Token, rhs_tree: Tree
    ) -> None:
        if generic_definitions_tree is not None:
            self._generic_typedef(type_name.value, generic_definitions_tree, rhs_tree)
        else:
            # Confusingly parsed but this is just a standard typedef
            self._specialized_typedef(type_name.value, [], rhs_tree)

    @inline_and_wrap_user_facing_errors("typedef<...>")
    def specialized_typedef(
        self, type_name: Token, specialization_tree: Tree, rhs_tree: Tree
    ) -> None:
        # Note: partial specializations are not allowed, generic mapping is empty
        specialization = [
            TypeTransformer.parse_specialization(tree)
            for tree in specialization_tree.children
        ]
        self._specialized_typedef(type_name.value, specialization, rhs_tree)


class ParseFunctionSignatures(Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self.current_file = None

        self._program = program
        self._function_body_trees: list[
            tuple[str, cg.Function, Tree, cg.GenericMapping]
        ] = []

    def parse_file(self, file_name: str, tree: Tree) -> None:
        self.current_file = file_name
        self.visit(tree)
        self.current_file = None

    def get_function_body_trees(
        self,
    ) -> list[tuple[str, cg.Function, Tree, cg.GenericMapping]]:
        return self._function_body_trees

    @inline_and_wrap_user_facing_errors("function[...] signature")
    def generic_named_function(
        self,
        generic_definitions_tree: Tree,
        generic_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        assert isinstance(generic_name, str)
        self._parse_generic_function_impl(
            generic_name,
            generic_definitions_tree,
            args_tree,
            return_type_tree,
            body_tree,
        )

    @inline_and_wrap_user_facing_errors("@operator[...] signature")
    def generic_operator_function(
        self,
        generic_definitions_tree: Tree,
        op_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        self._parse_generic_function_impl(
            op_name.value,
            generic_definitions_tree,
            args_tree,
            return_type_tree,
            body_tree,
        )

    def _parse_generic_function_impl(
        self,
        generic_name: str,
        generic_definitions_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        generic_mapping: cg.UnresolvedGenericMapping = {}
        generic_definitions: list[cg.GenericArgument] = []
        for generic_tree in generic_definitions_tree.children:
            generic = parse_generic_definition(generic_tree)
            generic_definitions.append(generic)

            if generic.is_value_arg:
                generic_mapping[generic.name] = cg.GenericValueReference(generic.name)
            else:
                generic_mapping[generic.name] = cg.UnresolvedGenericType(generic.name)

        arg_names: list[str] = []
        unresolved_return = TypeTransformer.parse(return_type_tree, generic_mapping)
        unresolved_args: list[cg.UnresolvedType] = []
        for name, type_tree in in_pairs(args_tree.children):
            unresolved_args.append(TypeTransformer.parse(type_tree, generic_mapping))
            arg_names.append(name.value)

        assert self.current_file is not None
        file_name = self.current_file

        def try_parse_fn_from_specialization(
            fn_name: str,
            specializations: list[cg.SpecializationItem],
        ) -> Optional[cg.Function]:
            assert generic_name == fn_name
            if len(generic_definitions) != len(specializations):
                return None  # SFINAE

            specialization_map = {
                generic.name: specialization
                for generic, specialization in zip(generic_definitions, specializations)
            }

            specialized_return = unresolved_return.produce_specialized_copy(
                specialization_map
            )
            specialized_args = [
                arg.produce_specialized_copy(specialization_map)
                for arg in unresolved_args
            ]

            try:
                function = self._build_function(
                    fn_name,
                    arg_names,
                    specialized_args,
                    specialized_return,
                    False,
                    specializations,
                )
            except SubstitutionFailure:
                return None  # SFINAE

            self._function_body_trees.append(
                (file_name, function, body_tree, specialization_map)
            )
            return function

        def try_deduce_specialization(
            fn_name: str, arguments: list[cg.TypedExpression]
        ) -> Optional[list[cg.SpecializationItem]]:
            assert generic_name == fn_name

            if len(arguments) != len(unresolved_args):
                return None  # SFINAE

            deduced_mapping: cg.UnresolvedGenericMapping = {}
            for actual_arg, unresolved_arg in zip(arguments, unresolved_args):
                if not unresolved_arg.pattern_match(
                    actual_arg.underlying_type, deduced_mapping
                ):
                    return None  # SFINAE

            # Convert the deduced mapping into a specialization
            deduced_specialization: list[cg.SpecializationItem] = []
            for generic in generic_definitions:
                specialization = deduced_mapping[generic.name]
                if isinstance(specialization, cg.CompileTimeConstant):
                    deduced_specialization.append(specialization.resolve())
                else:
                    resolved_type = self._program.resolve_type(specialization)
                    deduced_specialization.append(resolved_type)

            return deduced_specialization

        self._program.add_generic_function(
            generic_name,
            cg.GenericFunctionParser(
                generic_name,
                try_parse_fn_from_specialization,
                try_deduce_specialization,
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

        assert isinstance(name, Token)
        self._parse_function(name.value, args_tree, return_type_tree, body_tree, False)

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

        self._parse_function(
            op_name.value, args_tree, return_type_tree, body_tree, False
        )

    @inline_and_wrap_user_facing_errors("foreign signature")
    def foreign_function(
        self,
        fn_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
    ) -> None:
        self._parse_function(fn_name.value, args_tree, return_type_tree, None, True)

    def _parse_function(
        self,
        fn_name: str,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Optional[Tree],
        foreign: bool,
    ) -> None:
        arg_names: list[str] = []
        unresolved_return = TypeTransformer.parse(return_type_tree, {})
        unresolved_args: list[cg.UnresolvedType] = []
        for name, type_tree in in_pairs(args_tree.children):
            unresolved_args.append(TypeTransformer.parse(type_tree, {}))
            arg_names.append(name.value)

        func = self._build_function(
            fn_name, arg_names, unresolved_args, unresolved_return, foreign, []
        )
        self._program.add_function(func)

        # Save the body to parse later (TODO: maybe forward declarations
        # should be possible?)
        if body_tree is not None:
            assert self.current_file is not None
            self._function_body_trees.append((self.current_file, func, body_tree, {}))

    def _build_function(
        self,
        fn_name: str,
        arg_names: list[str],
        arg_unresolved_types: list[cg.UnresolvedType],
        unresolved_return_type: cg.UnresolvedType,
        foreign: bool,
        specialization: list[cg.SpecializationItem],
    ) -> cg.Function:
        fn_args: list[cg.Parameter] = []
        for arg_name, arg_unresolved_type in zip(arg_names, arg_unresolved_types):
            arg_type = self._program.resolve_type(arg_unresolved_type)
            fn_args.append(cg.Parameter(arg_name, arg_type))

            if arg_type.definition.is_void:
                raise VoidVariableDeclaration(
                    "argument", arg_name, arg_type.format_for_output_to_user()
                )

        fn_return_type = self._program.resolve_type(unresolved_return_type)
        fn_obj = cg.Function(fn_name, fn_args, fn_return_type, foreign, specialization)

        # main() must always return an int
        if (
            fn_obj.get_signature().is_main()
            and fn_obj.get_signature().return_type != cg.IntType()
        ):
            raise InvalidMainReturnType(
                fn_obj.get_signature().return_type.format_for_output_to_user()
            )

        return fn_obj


class ParseImports(Interpreter):
    def __init__(
        self,
        lark: Lark,
        program: cg.Program,
        fn_parser: ParseFunctionSignatures,
        include_path: list[Path],
        included_from: list[ResolvedPath],
        already_processed: set[ResolvedPath],
    ) -> None:
        super().__init__()

        self._lark = lark
        self._program = program
        self._fn_parser = fn_parser
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
            self._fn_parser,
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


def is_unresolved_specialization_list(
    lst: Any,
) -> TypeGuard[list[cg.UnresolvedSpecializationItem]]:
    # https://github.com/python/mypy/issues/3497#issuecomment-1083747764
    if not isinstance(lst, list):
        return False
    return all(isinstance(item, cg.UnresolvedSpecializationItem) for item in lst)


class ExpressionTransformer(Transformer):
    def __init__(
        self,
        program: cg.Program,
        function: cg.Function,
        scope: cg.Scope,
        generic_mapping: cg.GenericMapping,
    ) -> None:
        super().__init__(visit_tokens=True)

        self._program = program
        self._function = function
        self._scope = scope
        self._generic_mapping = generic_mapping

    @v_args(inline=True)
    def expression(self, value: FlattenedExpression) -> FlattenedExpression:
        assert isinstance(value, FlattenedExpression)
        return value

    def UNSIGNED_HEX_CONSTANT(self, value: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.UIntType(), value)
        return FlattenedExpression([const_expr])

    def SIGNED_INT(self, value: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.IntType(), value)
        return FlattenedExpression([const_expr])

    def BOOL_CONSTANT(self, value: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.BoolType(), value)
        return FlattenedExpression([const_expr])

    @v_args(inline=True)
    def compile_time_constant(self, constant: Token | FlattenedExpression) -> int:
        # TODO: this should be parsed correctly to begin with
        if isinstance(constant, FlattenedExpression):
            # TODO: this should be parsed correctly
            expression = constant.expression()
            assert len(constant.subexpressions) == 1
            assert isinstance(expression, cg.ConstantExpression)
            return int(expression.value)
        if isinstance(constant, Token):
            # TODO: user facing errors
            assert constant.value in self._generic_mapping
            mapped_value = self._generic_mapping[constant.value]
            assert isinstance(mapped_value, int)
            return mapped_value

        assert False

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
        fn_call_args = []

        for arg in fn_args:
            fn_call_args.append(arg.expression())
            flattened_expr.subexpressions.extend(arg.subexpressions)

        specialization: list[cg.SpecializationItem] = []
        if specialization_tree is not None:
            for specialization_item in specialization_tree.children:
                if isinstance(specialization_item, Tree):
                    assert specialization_item.data == "type"
                    specialization.append(
                        TypeTransformer.parse_and_resolve(
                            self._program,
                            specialization_item,
                            self._generic_mapping,
                        )
                    )
                else:
                    assert isinstance(specialization_item, int)
                    specialization.append(specialization_item)

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
        borrowed_this = this.add_parent(cg.BorrowExpression(this.expression(), False))

        fn_name, specialization_tree = name_tree.children
        assert isinstance(fn_name, str)

        fn_args = (borrowed_this, *args)
        assert is_flattened_expression_iterable(fn_args)

        return self._function_call_impl(fn_name, specialization_tree, fn_args)

    def ESCAPED_STRING(self, string: Token) -> FlattenedExpression:
        assert string[0] == '"' and string[-1] == '"'
        str_static_storage = self._program.add_static_string(string[1:-1])
        expr = FlattenedExpression([cg.VariableReference(str_static_storage)])

        # Implicitly take reference to string literal
        return expr.add_parent(cg.BorrowExpression(expr.expression(), True))

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
        borrow = cg.BorrowExpression(lhs.expression(), False)
        return lhs.add_parent(borrow)

    @v_args(inline=True)
    def const_borrow_operator_use(self, lhs: FlattenedExpression):
        borrow = cg.BorrowExpression(lhs.expression(), True)
        return lhs.add_parent(borrow)

    def struct_initializer_without_names(
        self, members: list[FlattenedExpression]
    ) -> FlattenedExpression:
        assert isinstance(members, list)

        member_exprs: list[cg.TypedExpression] = []
        combined_flattened = FlattenedExpression([])

        for item in members:
            combined_flattened.subexpressions.extend(item.subexpressions)
            member_exprs.append(item.expression())

        return combined_flattened.add_parent(cg.UnnamedInitializerList(member_exprs))

    def struct_initializer_with_names(
        self, member_with_names: list[FlattenedExpression | Token]
    ) -> FlattenedExpression:
        assert isinstance(member_with_names, list)

        # Use zip to transpose a list of pairs into a pair of lists.
        names, members = list(map(list, zip(*in_pairs(member_with_names))))

        member_exprs: list[cg.TypedExpression] = []
        combined_flattened = FlattenedExpression([])

        for item in members:
            combined_flattened.subexpressions.extend(item.subexpressions)
            member_exprs.append(item.expression())

        return combined_flattened.add_parent(
            cg.NamedInitializerList(member_exprs, names)
        )

    @v_args(inline=True)
    def adhoc_struct_initialization(
        self, expr: FlattenedExpression
    ) -> FlattenedExpression:
        assert isinstance(expr.expression(), cg.InitializerList)
        return expr

    @staticmethod
    def parse(
        program: cg.Program,
        function: cg.Function,
        scope: cg.Scope,
        generic_mapping: cg.GenericMapping,
        body: Tree,
    ) -> FlattenedExpression:
        result = ExpressionTransformer(
            program, function, scope, generic_mapping
        ).transform(body)
        assert isinstance(result, FlattenedExpression)
        return result


def generate_standalone_expression(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
) -> None:
    assert len(body.children) == 1
    flattened_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, body
    )

    scope.add_generatable(flattened_expr.subexpressions)


def generate_return_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
) -> None:
    if not body.children:
        expr = cg.ReturnStatement(cg.VoidType())
        scope.add_generatable(expr)
        return

    (expr,) = body.children
    flattened_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, expr
    )

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
    generic_mapping: cg.GenericMapping,
) -> None:
    var_name, type_tree, rhs_tree = body.children
    rhs: Optional[FlattenedExpression] = (
        ExpressionTransformer.parse(program, function, scope, generic_mapping, rhs_tree)
        if rhs_tree is not None
        else None
    )

    assert isinstance(var_name, Token)
    assert isinstance(type_tree, Tree)

    var_type = TypeTransformer.parse_and_resolve(program, type_tree, generic_mapping)
    if var_type.definition.is_void:
        raise VoidVariableDeclaration(
            "variable", var_name, var_type.format_for_output_to_user()
        )

    var = cg.StackVariable(var_name, var_type, is_const, rhs is not None)
    scope.add_variable(var)

    if rhs is None:
        return

    scope.add_generatable(rhs.subexpressions)
    scope.add_generatable(cg.VariableAssignment(var, rhs.expression()))


def generate_if_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
) -> None:
    condition_tree, if_scope_tree, else_scope_tree = body.children
    assert len(condition_tree.children) == 1

    condition_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, condition_tree
    )

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
    generic_mapping: cg.GenericMapping,
) -> None:
    condition_tree, inner_scope_tree = body.children
    assert len(condition_tree.children) == 1

    condition_expr = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, condition_tree
    )

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
    generic_mapping: cg.GenericMapping,
) -> None:

    lhs_tree, rhs_tree = body.children
    lhs = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, lhs_tree
    )
    rhs = ExpressionTransformer.parse(
        program, function, scope, generic_mapping, rhs_tree
    )

    scope.add_generatable(lhs.subexpressions)
    scope.add_generatable(rhs.subexpressions)
    scope.add_generatable(cg.Assignment(lhs.expression(), rhs.expression()))


def generate_scope_body(
    program: cg.Program,
    function: cg.Function,
    outer_scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
) -> None:
    inner_scope = cg.Scope(function.get_next_scope_id(), outer_scope)
    generate_body(program, function, inner_scope, body, generic_mapping)
    outer_scope.add_generatable(inner_scope)


def generate_body(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
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
    generic_mapping: cg.GenericMapping,
):
    generate_body(program, function, function.top_level_scope, body, generic_mapping)

    # We cannot omit the "ret" instruction from LLVM IR. If the function returns
    # void, then we can add it ourselves, otherwise the user needs to fix it.

    if not function.top_level_scope.is_return_guaranteed():
        if function.get_signature().return_type.definition.is_void:
            function.top_level_scope.add_generatable(cg.ReturnStatement(cg.VoidType()))
        else:
            raise MissingFunctionReturn(
                function.get_signature().user_facing_name,
                body.meta.end_line,
            )


def append_file_to_program(
    lark: Lark,
    program: cg.Program,
    function_parser: ParseFunctionSignatures,
    file_path: ResolvedPath,
    include_path: list[Path],
    included_from: list[ResolvedPath],
    already_processed: set[ResolvedPath],
    debug_compiler: bool = False,
) -> None:
    already_processed.add(file_path)

    try:
        tree = parse_file_into_lark_tree(lark, file_path)

        ParseImports(
            lark,
            program,
            function_parser,
            include_path + [Path(file_path).parent],
            included_from + [file_path],
            already_processed,
        ).visit(tree)
        # TODO: these stages can be combined if we require forward declaration
        # FIXME: allow recursive types
        ParseTypeDefinitions(program).visit(tree)
        program.resolve_all_types()

        function_parser.parse_file(file_path, tree)
    except ErrorWithLineInfo as exc:
        if debug_compiler:
            traceback.print_exc()
            print("~~~ User-facing error message ~~~")

        context_str = f", in '{exc.context}'" if exc.context else ""
        print(f"File '{file_path}', line {exc.line}{context_str}", file=sys.stderr)
        print(f"    {exc.message}", file=sys.stderr)

        if included_from:
            print(file=sys.stderr)

        for file in reversed(included_from):
            print(f"Included from file '{file}'", file=sys.stderr)

        sys.exit(1)


def parse_file_into_lark_tree(lark: Lark, file_path: ResolvedPath) -> Tree:
    with open(file_path, encoding="utf-8") as source_file:
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

        raise InvalidSyntax(error_message_context, exc.line)


def generate_ir_from_source(
    file_path: Path, include_path: list[Path], debug_compiler: bool = False
) -> str:
    grammar_path = Path(__file__).parent / "grammar.lark"
    lark = Lark.open(
        str(grammar_path), parser="lalr", start="program", propagate_positions=True
    )

    program = cg.Program()
    fn_parser = ParseFunctionSignatures(program)
    append_file_to_program(
        lark,
        program,
        fn_parser,
        ResolvedPath(file_path),
        include_path,
        [],
        set(),
        debug_compiler,
    )

    for file_name, fn, body, specialization in fn_parser.get_function_body_trees():
        generate_function_body(program, fn, body, specialization)

    return "\n".join(program.generate_ir())

import sys
import traceback
from dataclasses import dataclass
from functools import partial
from parser.lexer_parser import Interpreter, Token, Transformer, Tree, run_lexer_parser
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, TypeGuard

import codegen as cg
from codegen.user_facing_errors import (
    CircularImportException,
    ErrorWithLineInfo,
    ErrorWithLocationInfo,
    FailedLookupError,
    FileDoesNotExistException,
    FileIsAmbiguousException,
    GenericHasGenericAnnotation,
    GrapheneError,
    MissingFunctionReturn,
    RepeatedGenericName,
    SourceLocation,
    StructMemberRedeclaration,
    VoidVariableDeclaration,
)

UnresolvedGenericMapping = dict[str, cg.UnresolvedSpecializationItem]


class ResolvedPath(str):
    def __new__(cls, path: Path) -> "ResolvedPath":
        return str.__new__(cls, str(path.resolve()))


def in_pairs(iterable: Iterable) -> Iterable:
    # [iter(...), iter(...)] would make two different list_iterator objects.
    # We only want one.
    chunks = [iter(iterable)] * 2

    return zip(*chunks, strict=True)


def get_optional_children(tree: Optional[Tree]) -> tuple[Any]:
    if tree is None:
        return tuple()
    return tuple(tree.children)


def inline_arguments(func):
    def wrapper(obj, arg: Tree | Token):
        if isinstance(arg, Token):
            return func(obj, arg)
        return func(obj, *arg.children)

    return wrapper


def inline_and_wrap_user_facing_errors(context: str):
    def wrapper(func):
        def wrap_user_facing_errors(obj, arg: Tree | Token):
            try:
                return inline_arguments(func)(obj, arg)
            except GrapheneError as exc:
                raise ErrorWithLineInfo(
                    exc.message,
                    arg.meta.start.line,
                    context,
                ) from exc

        return wrap_user_facing_errors

    return wrapper


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


def parse_generic_definition(definition: Token) -> cg.GenericArgument:
    assert definition.name == "GENERIC_IDENTIFIER"
    return cg.GenericArgument(definition.value, definition.value[0] == "@")


class TypeTransformer(Transformer):
    def __init__(self, generic_args: UnresolvedGenericMapping) -> None:
        super().__init__(visit_tokens=True)
        self._generic_args = generic_args

    @inline_arguments
    def compile_time_constant(
        self, constant: cg.CompileTimeConstant
    ) -> cg.CompileTimeConstant:
        assert isinstance(constant, cg.CompileTimeConstant)
        return constant

    @inline_arguments
    def SIGNED_INT(self, token: Token) -> cg.CompileTimeConstant:
        return cg.NumericLiteralConstant(int(token.value))

    @inline_arguments
    def NUMERIC_GENERIC_IDENTIFIER(self, token: Token) -> cg.CompileTimeConstant:
        if not token.value in self._generic_args:
            raise FailedLookupError("numeric generic", f"[{token.value}, ...]")

        result = self._generic_args[token.value]
        assert isinstance(result, cg.CompileTimeConstant)  # TODO: user facing error
        return result

    @inline_arguments
    def type(self, child: cg.UnresolvedType) -> cg.UnresolvedType:
        assert isinstance(child, cg.UnresolvedType)
        return child

    @inline_arguments
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
        return cg.UnresolvedNamedType(name.value, tuple(specialization))

    @inline_arguments
    def ref_type(self, value_type: cg.UnresolvedType) -> cg.UnresolvedType:
        return cg.UnresolvedReferenceType(value_type)

    @inline_arguments
    def stack_array_type(
        self, element_type: cg.UnresolvedType, *dimensions: cg.CompileTimeConstant
    ) -> cg.UnresolvedType:
        return cg.UnresolvedStackArrayType(element_type, dimensions)

    @inline_arguments
    def heap_array_type(
        self, element_type: cg.UnresolvedType, *known_dimensions: cg.CompileTimeConstant
    ) -> cg.UnresolvedType:
        return cg.UnresolvedHeapArrayType(element_type, known_dimensions)

    @inline_arguments
    def struct_type(
        self, *member_trees: Token | cg.UnresolvedType
    ) -> cg.UnresolvedType:
        members: dict[str, cg.UnresolvedType] = {}

        for member_token, member_type in in_pairs(member_trees):
            assert isinstance(member_token, Token)
            assert isinstance(member_type, cg.UnresolvedType)
            member_name = member_token.value

            if member_name in members:
                raise StructMemberRedeclaration(
                    member_name,
                    members[member_name].format_for_output_to_user(),
                    member_token.meta.start.line,
                )

            members[member_name] = member_type

        # As of Python 3.7, dict preserves insertion order.
        return cg.UnresolvedStructType(tuple((n, t) for n, t in members.items()))

    @classmethod
    def parse_and_resolve(
        cls,
        program: cg.Program,
        tree: Tree,
        generic_mapping: cg.GenericMapping,
    ) -> cg.SpecializationItem:
        unresolved_mapping: UnresolvedGenericMapping = {}
        for key, item in generic_mapping.mapping.items():
            if isinstance(item, cg.Type):
                unresolved_mapping[key.name] = cg.UnresolvedTypeWrapper(item)
            else:
                assert isinstance(item, int)
                unresolved_mapping[key.name] = cg.NumericLiteralConstant(item)

        result = cls(unresolved_mapping).transform(tree)

        if isinstance(result, cg.CompileTimeConstant):
            return result.resolve()

        assert isinstance(result, cg.UnresolvedType)
        return program.symbol_table.resolve_type(result)

    @classmethod
    def parse(
        cls, tree: Tree, generic_mapping: UnresolvedGenericMapping
    ) -> cg.UnresolvedType:
        result = cls(generic_mapping).transform(tree)
        assert isinstance(result, cg.UnresolvedType)
        return result

    @classmethod
    def parse_specialization(cls, tree: Tree) -> cg.UnresolvedSpecializationItem:
        result = cls({}).transform(tree)
        assert isinstance(result, cg.UnresolvedSpecializationItem)
        return result


class ParseTypeDefinitions(Interpreter):
    def __init__(
        self, file: str, include_hierarchy: list[str], program: cg.Program
    ) -> None:
        super().__init__()

        self._program = program
        self._file = file
        self._include_hierarchy = include_hierarchy

    def _generic_typedef(
        self, prefix: Token, generic_definitions_tree: Tree, rhs_tree: Tree
    ) -> None:
        assert prefix.meta.start.line is not None
        loc = SourceLocation(
            prefix.meta.start.line, self._file, tuple(self._include_hierarchy)
        )

        generic_mapping: UnresolvedGenericMapping = {}
        generic_definitions: list[cg.GenericArgument] = []

        for generic_tree in generic_definitions_tree.children:
            assert isinstance(generic_tree, Token)
            definition = parse_generic_definition(generic_tree)
            generic_definitions.append(definition)

            if definition.name in generic_mapping:
                assert generic_tree.meta.start.line is not None
                raise RepeatedGenericName(
                    generic_tree.value,
                    generic_tree.meta.start.line,
                    prefix.value,
                )

            generic_mapping[definition.name] = (
                cg.GenericValueReference(definition.name)
                if definition.is_value_arg
                else cg.UnresolvedGenericType(definition.name)
            )

        rhs_type = TypeTransformer.parse(rhs_tree, generic_mapping)
        self._program.symbol_table.add_type(
            cg.Typedef.construct(
                prefix.value, tuple(generic_definitions), tuple(), rhs_type, loc
            )
        )

    def _specialized_typedef(
        self,
        prefix: Token,
        specialization: list[cg.UnresolvedSpecializationItem],
        rhs_tree: Tree,
    ) -> None:
        assert prefix.meta.start.line is not None
        loc = SourceLocation(
            prefix.meta.start.line, self._file, tuple(self._include_hierarchy)
        )

        rhs_type = TypeTransformer.parse(rhs_tree, {})
        self._program.symbol_table.add_type(
            cg.Typedef.construct(
                prefix.value, tuple(), tuple(specialization), rhs_type, loc
            )
        )

    @inline_and_wrap_user_facing_errors("typedef")
    def generic_typedef(
        self, generic_definitions_tree: Optional[Tree], type_name: Token, rhs_tree: Tree
    ) -> None:
        if generic_definitions_tree is not None:
            self._generic_typedef(type_name, generic_definitions_tree, rhs_tree)
        else:
            # Confusingly parsed but this is just a standard typedef
            self._specialized_typedef(type_name, [], rhs_tree)

    @inline_and_wrap_user_facing_errors("typedef<...>")
    def specialized_typedef(
        self, type_name: Token, specialization_tree: Tree, rhs_tree: Tree
    ) -> None:
        # Note: partial specializations are not allowed, generic mapping is empty
        specialization = []
        for tree in specialization_tree.children:
            assert isinstance(tree, Tree)
            specialization.append(TypeTransformer.parse_specialization(tree))

        self._specialized_typedef(type_name, specialization, rhs_tree)


class ParseFunctionSignatures(Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._current_file: Optional[str] = None
        self._include_hierarchy: Optional[list[str]] = None

        self._program = program
        self._function_mapping: dict[cg.FunctionDeclaration, Tree] = {}

    def parse_file(
        self, file_name: str, include_hierarchy: list[str], tree: Tree
    ) -> None:
        self._current_file = file_name
        self._include_hierarchy = include_hierarchy
        self.visit(tree)
        self._current_file = None
        self._include_hierarchy = None

    def get_functions_to_codegen(
        self,
    ) -> Generator[
        tuple[
            cg.FunctionDeclaration,
            cg.FunctionSignature,
            cg.GenericMapping,
            Optional[Tree],
        ],
        None,
        None,
    ]:
        while to_generate := self._program.symbol_table.get_next_function_to_codegen():
            declaration, signature = to_generate

            mapping = cg.GenericMapping({}, [])
            body_tree = None
            if not declaration.is_foreign:
                declaration.pattern_match(
                    signature.arguments, signature.specialization, mapping
                )
                body_tree = self._function_mapping[declaration]

            yield declaration, signature, mapping, body_tree

    @inline_and_wrap_user_facing_errors("function[...] signature")
    def generic_named_function(
        self,
        generic_definitions_tree: Tree,
        generic_token: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        assert isinstance(generic_token, Token)
        self._parse_generic_function_impl(
            generic_token.value,
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

    @inline_and_wrap_user_facing_errors("@assignment[...] signature")
    def generic_assignment_function(
        self,
        generic_definitions_tree: Tree,
        assignment_op_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        self._parse_generic_function_impl(
            assignment_op_name.value,
            generic_definitions_tree,
            args_tree,
            return_type_tree,
            body_tree,
        )

    @inline_and_wrap_user_facing_errors("@implicit[...] signature")
    def generic_implicit_function(
        self,
        generic_definitions_tree: Tree,
        implicit_function_name: Token,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        # TODO: user facing error
        assert implicit_function_name.value in ("has_next", "get_next", "destruct")
        self._parse_generic_function_impl(
            "__builtin_" + implicit_function_name.value,
            generic_definitions_tree,
            args_tree,
            return_type_tree,
            body_tree,
        )

    def _parse_generic_function_impl(
        self,
        fn_name: str,
        generic_definitions_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ):
        assert self._current_file is not None
        assert self._include_hierarchy is not None
        # Save these for later.
        current_file = self._current_file
        include_hierarchy = self._include_hierarchy

        location = SourceLocation(
            args_tree.meta.start.line,
            self._current_file,
            tuple(self._include_hierarchy),
        )

        generic_mapping: UnresolvedGenericMapping = {}
        generic_definitions: list[cg.GenericArgument] = []

        # One for the type and one for the actual args.
        variadic_type_pack_name: Optional[str] = None
        variadic_args_pack_name: Optional[str] = None

        for generic_token in generic_definitions_tree.children:
            assert isinstance(generic_token, Token)
            if generic_token.name == "GENERIC_IDENTIFIER":
                generic = parse_generic_definition(generic_token)
                generic_definitions.append(generic)

                if generic.is_value_arg:
                    generic_mapping[generic.name] = cg.GenericValueReference(
                        generic.name
                    )
                else:
                    generic_mapping[generic.name] = cg.UnresolvedGenericType(
                        generic.name
                    )
            else:
                assert generic_token.name == "GENERIC_PACK_IDENTIFIER"
                # NOTE enforced by the grammar. We might want to relax this
                # later and allow more than one pack per function.
                assert variadic_type_pack_name is None
                variadic_type_pack_name = generic_token.value

        unresolved_return = TypeTransformer.parse(return_type_tree, generic_mapping)

        arg_names: list[str] = []
        unresolved_args: list[cg.UnresolvedType] = []
        for name, type_tree in in_pairs(args_tree.children):
            assert isinstance(name, Token)
            # TODO refactor, this is very ugly.
            if isinstance(type_tree, Token):
                # TODO user-facing error.
                assert type_tree.value == variadic_type_pack_name
                variadic_args_pack_name = f"{name.value}..."
            else:
                assert isinstance(type_tree, Tree)
                arg_names.append(name.value)

                unresolved_args.append(
                    TypeTransformer.parse(type_tree, generic_mapping)
                )

        fn_declaration = cg.FunctionDeclaration.construct(
            fn_name,
            False,
            tuple(generic_definitions),
            [],
            tuple(arg_names),
            variadic_args_pack_name,
            tuple(unresolved_args),
            variadic_type_pack_name,
            unresolved_return,
            location,
        )
        self._function_mapping[fn_declaration] = body_tree
        self._program.symbol_table.add_function(fn_declaration)

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
        specialization_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        if specialization_tree is not None:
            raise NotImplementedError()

        self._parse_function(
            op_name.value, args_tree, return_type_tree, body_tree, False
        )

    @inline_and_wrap_user_facing_errors("@assignment signature")
    def specialized_assignment_function(
        self,
        assignment_op_name: Token,
        specialization_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        if specialization_tree is not None:
            raise NotImplementedError()

        self._parse_function(
            assignment_op_name.value, args_tree, return_type_tree, body_tree, False
        )

    @inline_and_wrap_user_facing_errors("@implicit signature")
    def specialized_implicit_function(
        self,
        implicit_function_name: Token,
        specialization_tree: Tree,
        args_tree: Tree,
        return_type_tree: Tree,
        body_tree: Tree,
    ) -> None:
        if specialization_tree is not None:
            raise NotImplementedError()

        # TODO: user facing error
        assert implicit_function_name.value in ("has_next", "get_next", "destruct")

        self._parse_function(
            "__builtin_" + implicit_function_name.value,
            args_tree,
            return_type_tree,
            body_tree,
            False,
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
        assert self._current_file is not None
        assert self._include_hierarchy is not None

        location = SourceLocation(
            args_tree.meta.start.line,
            self._current_file,
            tuple(self._include_hierarchy),
        )

        unresolved_return = TypeTransformer.parse(return_type_tree, {})

        arg_names: list[str] = []
        unresolved_args: list[cg.UnresolvedType] = []
        for name, type_tree in in_pairs(args_tree.children):
            unresolved_args.append(TypeTransformer.parse(type_tree, {}))
            arg_names.append(name.value)

        func = cg.FunctionDeclaration.construct(
            fn_name,
            foreign,
            tuple(),
            [],
            tuple(arg_names),
            None,
            tuple(unresolved_args),
            None,
            unresolved_return,
            location,
        )
        if not foreign:
            assert body_tree is not None
            self._function_mapping[func] = body_tree

        self._program.symbol_table.add_function(func)


class ParseImports(Interpreter):
    def __init__(
        self,
        program: cg.Program,
        fn_parser: ParseFunctionSignatures,
        include_path: list[Path],
        included_from: list[ResolvedPath],
        already_processed: set[ResolvedPath],
    ) -> None:
        super().__init__()

        self._program = program
        self._fn_parser = fn_parser
        self._include_path = include_path
        self._included_from = included_from
        self._already_processed = already_processed

    def require_once(self, path_tree: Tree) -> None:
        path_token = path_tree.children[0]
        assert isinstance(path_token, Token)

        try:
            self._require_once_impl(path_token.value[1:-1])
        except GrapheneError as exc:
            raise ErrorWithLineInfo(
                exc.message,
                path_tree.meta.start.line,
                f"@require_once {path_token.value}",
            ) from exc

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


def is_tree_or_token_iterable(
    exprs: Iterable[Any],
) -> TypeGuard[Iterable[Tree | Token]]:
    # https://github.com/python/mypy/issues/3497#issuecomment-1083747764
    return all(isinstance(expr, (Tree, Token)) for expr in exprs)


def is_unresolved_specialization_list(
    lst: Any,
) -> TypeGuard[Iterable[cg.UnresolvedSpecializationItem]]:
    # https://github.com/python/mypy/issues/3497#issuecomment-1083747764
    if not isinstance(lst, Iterable):
        return False
    return all(isinstance(item, cg.UnresolvedSpecializationItem) for item in lst)


class ExpressionParser(Interpreter):
    def __init__(
        self,
        program: cg.Program,
        function: cg.Function,
        scope: cg.Scope,
        generic_mapping: cg.GenericMapping,
    ) -> None:
        super().__init__()

        self._program = program
        self._function = function
        self._scope = scope
        self._generic_mapping = generic_mapping

    def visit(self, tree: Tree | Token) -> FlattenedExpression:
        result = super().visit(tree)
        assert isinstance(result, FlattenedExpression)
        return result

    @inline_arguments
    def expression(self, expr_tree: Tree) -> FlattenedExpression:
        return self.visit(expr_tree)

    def UNSIGNED_HEX_CONSTANT(self, token: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.UIntType(), token.value)
        return FlattenedExpression([const_expr])

    def NUMERIC_GENERIC_IDENTIFIER(self, token: Token) -> FlattenedExpression:
        argument = cg.GenericArgument(token.value, True)
        if argument not in self._generic_mapping.mapping:
            raise FailedLookupError("numeric generic", f"[{token.value}, ...]")

        mapped_value = self._generic_mapping.mapping[argument]
        assert isinstance(mapped_value, int)  # TODO: user facing error
        const_expr = cg.ConstantExpression(cg.IntType(), str(mapped_value))
        return FlattenedExpression([const_expr])

    def SIGNED_INT(self, token: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.IntType(), token.value)
        return FlattenedExpression([const_expr])

    def BOOL_CONSTANT(self, token: Token) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.BoolType(), token.value)
        return FlattenedExpression([const_expr])

    @inline_arguments
    def operator_use(
        self, lhs_tree: Tree, operator: Token, rhs_tree: Tree
    ) -> FlattenedExpression:
        lhs = self.visit(lhs_tree)
        rhs = self.visit(rhs_tree)
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

    @inline_arguments
    def unary_operator_use(
        self, operator: Token, rhs_tree: Tree
    ) -> FlattenedExpression:
        rhs = self.visit(rhs_tree)
        assert isinstance(rhs, FlattenedExpression)

        flattened_expr = FlattenedExpression(rhs.subexpressions)

        call_expr = self._program.lookup_call_expression(
            operator.value,
            [],  # Don't specialize operators
            [rhs.expression()],
        )
        return flattened_expr.add_parent(call_expr)

    @inline_arguments
    def logical_operator_use(
        self, lhs_tree: Tree, operator: Token, rhs_tree: Tree
    ) -> FlattenedExpression:
        assert operator.value in ("and", "or")

        lhs = self.visit(lhs_tree)
        rhs = self.visit(rhs_tree)
        assert isinstance(lhs, FlattenedExpression)
        assert isinstance(rhs, FlattenedExpression)

        flattened_expr = FlattenedExpression([])
        flattened_expr.subexpressions.extend(lhs.subexpressions)
        flattened_expr.add_parent(
            cg.LogicalOperator(
                operator.value,
                self._function.get_next_label_id(),
                lhs.expression(),
                rhs.subexpressions[:-1],
                rhs.expression(),
            )
        )

        return flattened_expr

    def _function_call_inner(
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
        for specialization_item_tree in get_optional_children(specialization_tree):
            specialization.append(
                TypeTransformer.parse_and_resolve(
                    self._program,
                    specialization_item_tree,
                    self._generic_mapping,
                )
            )

        call_expr = self._program.lookup_call_expression(
            fn_name, specialization, fn_call_args
        )
        return flattened_expr.add_parent(call_expr)

    @staticmethod
    def _split_function_call_args(
        *args: Any,
    ) -> tuple[list[Tree | Token], Optional[Token]]:
        *arg_trees, pack_name = args

        assert is_tree_or_token_iterable(arg_trees)
        if pack_name is not None:
            assert isinstance(pack_name, Token)

        return [*arg_trees], pack_name

    def _function_call_outer(
        self, name_tree: Tree, is_ufcs_call: bool, *all_args: Any
    ) -> FlattenedExpression:
        fn_name, specialization_tree = name_tree.children
        assert isinstance(fn_name, Token)

        arg_trees, pack_name = self._split_function_call_args(*all_args)

        args = [self.visit(tree) for tree in arg_trees]

        if pack_name:
            variadic_vars = self._scope.search_for_generic_pack(pack_name.value)
            args.extend(
                FlattenedExpression([cg.VariableReference(var)])
                for var in variadic_vars
            )

        # TODO perhaps we shouldn't always borrow this, although this is a bit
        # tricky as we haven't done overload resolution yet (which depends on
        # whether we borrow or not). A solution would be to borrow if we can,
        # otherwise pass an unborrowed/const-reference and let overload
        # resolution figure it out, although this isn't very explicit.
        if is_ufcs_call:
            this_arg = args[0]
            if this_arg.expression().is_indirect_pointer_to_type:
                this_arg.add_parent(cg.BorrowExpression(this_arg.expression(), False))

        assert isinstance(fn_name, Token)
        assert is_flattened_expression_iterable(args)

        return self._function_call_inner(fn_name.value, specialization_tree, args)

    @inline_arguments
    def function_call(self, name_tree: Tree, *all_args: Any) -> FlattenedExpression:
        return self._function_call_outer(name_tree, False, *all_args)

    @inline_arguments
    def ufcs_call(
        self, this_tree: Tree, name_tree: Tree, *all_args: Any
    ) -> FlattenedExpression:
        return self._function_call_outer(name_tree, True, this_tree, *all_args)

    def ESCAPED_STRING(self, string_token: Token) -> FlattenedExpression:
        string = string_token.value
        assert string[0] == '"' and string[-1] == '"'
        str_static_storage = self._program.add_static_string(string[1:-1])
        expr = FlattenedExpression([cg.VariableReference(str_static_storage)])

        # Implicitly take reference to string literal
        return expr.add_parent(cg.BorrowExpression(expr.expression(), True))

    @inline_arguments
    def accessed_variable_name(self, var_name: Token) -> FlattenedExpression:
        var = self._scope.search_for_variable(var_name.value)
        if var is None:
            raise FailedLookupError("variable", var_name.value)

        return FlattenedExpression([cg.VariableReference(var)])

    def _ensure_pointer_is_available(
        self, expr: FlattenedExpression
    ) -> FlattenedExpression:
        # Copy expression to stack if it is not a pointer
        if expr.expression().has_address:
            return expr

        temp_var = cg.StackVariable("", expr.type(), True, True)  # FIXME
        self._scope.add_variable(temp_var)

        expr.subexpressions.append(cg.VariableAssignment(temp_var, expr.expression()))
        return expr.add_parent(cg.VariableReference(temp_var))

    @inline_arguments
    def array_index_access(
        self, lhs_tree: Tree, *index_expr_trees: Tree
    ) -> FlattenedExpression:
        index_exprs: list[FlattenedExpression] = [
            self.visit(tree) for tree in index_expr_trees
        ]
        lhs = self._ensure_pointer_is_available(self.visit(lhs_tree))
        lhs_expr = lhs.expression()

        cg_indices: list[cg.TypedExpression] = []
        for index_expr in index_exprs:
            cg_indices.append(index_expr.expression())
            lhs.subexpressions.extend(index_expr.subexpressions)

        return lhs.add_parent(cg.ArrayIndexAccess(lhs_expr, cg_indices))

    @inline_arguments
    def struct_member_access(
        self, lhs_tree: Tree, member_name: Token
    ) -> FlattenedExpression:
        lhs = self.visit(lhs_tree)
        assert isinstance(member_name, Token)

        struct_access = cg.StructMemberAccess(lhs.expression(), member_name.value)
        return lhs.add_parent(struct_access)

    @inline_arguments
    def borrow_operator_use(self, lhs_tree: Tree) -> FlattenedExpression:
        lhs = self.visit(lhs_tree)
        borrow = cg.BorrowExpression(lhs.expression(), False)
        return lhs.add_parent(borrow)

    @inline_arguments
    def const_borrow_operator_use(self, lhs_tree: Tree) -> FlattenedExpression:
        lhs = self.visit(lhs_tree)
        borrow = cg.BorrowExpression(lhs.expression(), True)
        return lhs.add_parent(borrow)

    def struct_initializer_without_names(
        self, member_trees: Tree
    ) -> FlattenedExpression:
        members = []
        for member_tree in member_trees.children:
            assert member_tree is not None
            members.append(self.visit(member_tree))

        member_exprs: list[cg.TypedExpression] = []
        combined_flattened = FlattenedExpression([])

        for item in members:
            combined_flattened.subexpressions.extend(item.subexpressions)
            member_exprs.append(item.expression())

        return combined_flattened.add_parent(cg.UnnamedInitializerList(member_exprs))

    def struct_initializer_with_names(
        self, member_with_names: Tree
    ) -> FlattenedExpression:
        names: list[str] = []
        members: list[FlattenedExpression] = []
        for name_token, member_tree in in_pairs(member_with_names.children):
            assert isinstance(name_token, Token)
            assert isinstance(member_tree, (Tree | Token))
            names.append(name_token.value)
            members.append(self.visit(member_tree))

        member_exprs: list[cg.TypedExpression] = []
        combined_flattened = FlattenedExpression([])

        for item in members:
            combined_flattened.subexpressions.extend(item.subexpressions)
            member_exprs.append(item.expression())

        return combined_flattened.add_parent(
            cg.NamedInitializerList(member_exprs, names)
        )

    @inline_arguments
    def adhoc_struct_initialization(self, expr_tree: Tree) -> FlattenedExpression:
        return self.visit(expr_tree)

    @staticmethod
    def parse(
        program: cg.Program,
        function: cg.Function,
        scope: cg.Scope,
        generic_mapping: cg.GenericMapping,
        body: Tree,
    ) -> FlattenedExpression:
        result = ExpressionParser(program, function, scope, generic_mapping).visit(body)
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
    flattened_expr = ExpressionParser.parse(
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
    (expr,) = body.children
    if expr is None:
        scope.add_generatable(cg.ReturnStatement(cg.VoidType()))
        return

    assert isinstance(expr, Tree)
    flattened_expr = ExpressionParser.parse(
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
    assert isinstance(var_name, Token)
    assert isinstance(type_tree, Tree)

    if rhs_tree is not None:
        assert isinstance(rhs_tree, Tree)
        rhs = ExpressionParser.parse(
            program, function, scope, generic_mapping, rhs_tree
        )
    else:
        rhs = None

    var_type = TypeTransformer.parse_and_resolve(program, type_tree, generic_mapping)
    assert isinstance(var_type, cg.Type)
    if var_type.definition.is_void:
        raise VoidVariableDeclaration(
            "variable", var_name.value, var_type.format_for_output_to_user()
        )

    var = cg.StackVariable(var_name.value, var_type, is_const, rhs is not None)
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
    assert isinstance(condition_tree, Tree)
    assert isinstance(if_scope_tree, Tree)
    assert len(condition_tree.children) == 1

    condition_expr = ExpressionParser.parse(
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
        assert isinstance(else_scope_tree, Tree)
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
    assert isinstance(condition_tree, Tree)
    assert isinstance(inner_scope_tree, Tree)
    assert len(condition_tree.children) == 1

    condition_expr = ExpressionParser.parse(
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


def generate_for_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
) -> None:
    (
        variable_name,
        iterator_expression_tree,
        inner_scope_tree,
    ) = body.children
    assert isinstance(variable_name, Token)
    assert isinstance(inner_scope_tree, Tree)
    assert isinstance(iterator_expression_tree, Tree)

    # Outer scope
    outer_scope = cg.Scope(function.get_next_scope_id(), scope)

    #    Produce the iterator
    iter_expr = ExpressionParser.parse(
        program, function, scope, generic_mapping, iterator_expression_tree
    )
    #    Save the iterator onto the stack (for referencing)
    iter_variable = cg.StackVariable(
        f"__for_iter_{outer_scope.id}", iter_expr.type(), False, True
    )
    outer_scope.add_variable(iter_variable)
    outer_scope.add_generatable(iter_expr.subexpressions)
    outer_scope.add_generatable(
        cg.VariableAssignment(iter_variable, iter_expr.expression())
    )
    var_ref = cg.VariableReference(iter_variable)
    borrowed_iter_expr = cg.BorrowExpression(var_ref, False)
    outer_scope.add_generatable([var_ref, borrowed_iter_expr])

    # Inner scope
    inner_scope = cg.Scope(function.get_next_scope_id(), outer_scope)

    has_next_expr = program.lookup_call_expression(
        "__builtin_has_next", [], [borrowed_iter_expr]
    )
    get_next_expr = program.lookup_call_expression(
        "__builtin_get_next", [], [borrowed_iter_expr]
    )
    inner_scope.add_generatable(get_next_expr)

    iter_result_variable = cg.StackVariable(
        variable_name.value, get_next_expr.underlying_type, False, True
    )
    inner_scope.add_variable(iter_result_variable)
    inner_scope.add_generatable(
        cg.VariableAssignment(iter_result_variable, get_next_expr)
    )

    # For loop is just syntax sugar for a while loop
    generate_body(program, function, inner_scope, inner_scope_tree, generic_mapping)

    outer_scope.add_generatable(
        cg.WhileStatement(inner_scope.id, has_next_expr, [has_next_expr], inner_scope)
    )
    scope.add_generatable(outer_scope)


def generate_assignment(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: Tree,
    generic_mapping: cg.GenericMapping,
) -> None:
    lhs_tree, operator, rhs_tree = body.children
    assert isinstance(operator, Token)
    assert isinstance(lhs_tree, Tree)
    assert isinstance(rhs_tree, Tree)

    lhs = ExpressionParser.parse(program, function, scope, generic_mapping, lhs_tree)
    rhs = ExpressionParser.parse(program, function, scope, generic_mapping, rhs_tree)

    scope.add_generatable(lhs.subexpressions)
    scope.add_generatable(rhs.subexpressions)

    if operator.value == "=":
        scope.add_generatable(cg.Assignment(lhs.expression(), rhs.expression()))
    else:
        borrowed_lhs = cg.BorrowExpression(lhs.expression(), False)
        scope.add_generatable(borrowed_lhs)
        scope.add_generatable(
            program.lookup_call_expression(
                operator.value, [], [borrowed_lhs, rhs.expression()]
            )
        )


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
        "for_statement": generate_for_statement,
        "if_statement": generate_if_statement,
        "return_statement": generate_return_statement,
        "scope": generate_scope_body,
        "variable_declaration": partial(generate_variable_declaration, False),
        "while_statement": generate_while_statement,
    }

    for line in body.children:
        assert line is not None
        try:
            generators[line.name](program, function, scope, line, generic_mapping)
        except GrapheneError as exc:
            # TODO: more granular error messages
            raise ErrorWithLineInfo(
                exc.message,
                line.meta.start.line,
                function.get_signature().user_facing_name,
            ) from exc


def generate_function(
    program: cg.Program,
    declaration: cg.FunctionDeclaration,
    signature: cg.FunctionSignature,
    body: Optional[Tree],
    generic_mapping: cg.GenericMapping,
) -> None:
    try:
        fn = cg.Function(declaration.arg_names, signature, declaration.pack_type_name)
    except GrapheneError as e:
        assert isinstance(declaration.loc, SourceLocation)
        raise ErrorWithLineInfo(e.message, declaration.loc.line, "function declaration")

    if body is None:
        assert declaration.is_foreign
        return

    generate_body(program, fn, fn.top_level_scope, body, generic_mapping)

    # We cannot omit the "ret" instruction from LLVM IR. If the function returns
    # void, then we can add it ourselves, otherwise the user needs to fix it.
    if not fn.top_level_scope.is_return_guaranteed():
        if fn.get_signature().return_type.definition.is_void:
            fn.top_level_scope.add_generatable(cg.ReturnStatement(cg.VoidType()))
        else:
            raise MissingFunctionReturn(
                fn.get_signature().user_facing_name,
                body.meta.end.line,
            )

    program.add_function_body(fn)


def append_file_to_program(
    program: cg.Program,
    function_parser: ParseFunctionSignatures,
    file_path: ResolvedPath,
    include_path: list[Path],
    included_from: list[ResolvedPath],
    already_processed: set[ResolvedPath],
) -> None:
    already_processed.add(file_path)

    try:
        tree = parse_file_into_lark_tree(file_path)

        ParseImports(
            program,
            function_parser,
            include_path + [Path(file_path).parent],
            included_from + [file_path],
            already_processed,
        ).visit(tree)
        # TODO: these stages can be combined if we require forward declaration
        ParseTypeDefinitions(
            str(file_path), list(map(str, included_from)), program
        ).visit(tree)

        function_parser.parse_file(str(file_path), list(map(str, included_from)), tree)
        program.symbol_table.resolve_all_non_generics()
    except ErrorWithLineInfo as exc:
        raise ErrorWithLocationInfo(
            exc.message,
            SourceLocation(exc.line, str(file_path), tuple(map(str, included_from))),
            exc.context,
        ) from exc


def parse_file_into_lark_tree(file_path: ResolvedPath) -> Tree:
    # TODO: handle errors
    return run_lexer_parser(Path(file_path))


def generate_ir_from_source(
    file_path: Path, include_path: list[Path], debug_compiler: bool = False
) -> str:
    program = cg.Program()
    try:
        # Initial pass resolves all builtin types
        program.symbol_table.resolve_all_non_generics()

        fn_parser = ParseFunctionSignatures(program)
        append_file_to_program(
            program,
            fn_parser,
            ResolvedPath(file_path),
            include_path,
            [],
            set(),
        )

        for (
            declaration,
            signature,
            mapping,
            body,
        ) in fn_parser.get_functions_to_codegen():
            try:
                generate_function(program, declaration, signature, body, mapping)
            except ErrorWithLineInfo as exc:
                assert isinstance(declaration.loc, SourceLocation)
                location = SourceLocation(
                    exc.line, declaration.loc.file, declaration.loc.include_hierarchy
                )
                raise ErrorWithLocationInfo(exc.message, location, exc.context) from exc

    except ErrorWithLocationInfo as exc:
        if debug_compiler:
            traceback.print_exc(file=sys.stderr)
            print("~~~ User-facing error message ~~~", file=sys.stderr)

        context = f", in '{exc.context}'" if exc.context else ""
        print(f"{exc.loc}{context}", file=sys.stderr)
        print(f"    {exc.message}", file=sys.stderr)

        if isinstance(exc.loc, SourceLocation):
            if exc.loc.include_hierarchy:
                print(file=sys.stderr)

            for file in reversed(exc.loc.include_hierarchy):
                print(f"Included from file '{file}'", file=sys.stderr)

        sys.exit(1)

    return "\n".join(program.generate_ir())

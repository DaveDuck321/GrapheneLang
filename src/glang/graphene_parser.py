import sys
import traceback
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard
from uuid import UUID

from glang import codegen as cg
from glang.codegen.user_facing_errors import (
    BorrowWithNoAddressError,
    CircularImportException,
    ErrorWithLineInfo,
    ErrorWithLocationInfo,
    FailedLookupError,
    FileDoesNotExistException,
    FileIsAmbiguousException,
    GenericHasGenericAnnotation,
    GrapheneError,
    MissingFunctionReturn,
    NotInLoopStatementError,
    RepeatedGenericName,
    SourceLocation,
    StructMemberRedeclaration,
    VoidVariableDeclaration,
)
from glang.parser import lexer_parser as lp

UnresolvedGenericMapping = dict[str, cg.UnresolvedSpecializationItem]


class ResolvedPath(str):
    # See https://docs.astral.sh/ruff/rules/no-slots-in-str-subclass/.
    __slots__ = ()

    def __new__(cls, path: Path) -> "ResolvedPath":
        return str.__new__(cls, str(path.resolve()))


def search_include_path_for_file(
    relative_file_path: str, include_path: list[Path]
) -> ResolvedPath | None:
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


def parse_generic_definition(node: lp.GenericDefinition) -> cg.GenericArgument:
    return cg.GenericArgument(node.name, isinstance(node, lp.NumericGenericDefinition))


class TypeParser(lp.Interpreter):
    def __init__(self, generic_args: UnresolvedGenericMapping) -> None:
        super().__init__()

        self._generic_args = generic_args

    def parse_type(self, node: lp.Type) -> cg.UnresolvedType:
        try:
            return self.parse(node, cg.UnresolvedType)
        except GrapheneError as e:
            raise ErrorWithLineInfo(e.message, node.meta.start.line) from e

    def parse_specialization(
        self, node: lp.Type | lp.CompileTimeConstant
    ) -> cg.UnresolvedSpecializationItem:
        return self.parse(node, cg.UnresolvedSpecializationItem)

    def parse_constant(self, node: lp.CompileTimeConstant) -> cg.CompileTimeConstant:
        return self.parse(node, cg.CompileTimeConstant)

    def NumericGenericIdentifier(
        self, node: lp.NumericGenericIdentifier
    ) -> cg.CompileTimeConstant:
        if node.value not in self._generic_args:
            raise FailedLookupError("numeric generic", f"[{node.value}, ...]")

        result = self._generic_args[node.value]
        assert isinstance(result, cg.CompileTimeConstant)
        return result

    def NumericIdentifier(self, node: lp.NumericIdentifier) -> cg.CompileTimeConstant:
        return cg.NumericLiteralConstant(node.value)

    def NamedType(self, node: lp.NamedType) -> cg.UnresolvedType:
        if node.name in self._generic_args:
            if len(node.specialization) != 0:
                raise GenericHasGenericAnnotation(node.name)

            result = self._generic_args[node.name]
            assert isinstance(result, cg.UnresolvedType)
            return result

        return cg.UnresolvedNamedType(
            node.name,
            tuple(self.parse_specialization(item) for item in node.specialization),
        )

    def ReferenceType(self, node: lp.ReferenceType) -> cg.UnresolvedType:
        return cg.UnresolvedReferenceType(self.parse_type(node.value_type), node.is_mut)

    def StackArrayType(self, node: lp.StackArrayType) -> cg.UnresolvedType:
        return cg.UnresolvedStackArrayType(
            self.parse_type(node.base_type),
            tuple(self.parse_constant(item) for item in node.size),
        )

    def HeapArrayType(self, node: lp.HeapArrayType) -> cg.UnresolvedType:
        return cg.UnresolvedHeapArrayType(
            self.parse_type(node.base_type),
            tuple(self.parse_constant(item) for item in node.size),
            node.is_mut,
        )

    def StructType(self, node: lp.StructType) -> cg.UnresolvedType:
        all_members: dict[str, cg.UnresolvedType] = {}
        for member_name, member_type in node.members:
            if member_name in all_members:
                # TODO: add meta to member_name
                raise StructMemberRedeclaration(
                    member_name,
                    all_members[member_name].format_for_output_to_user(),
                    member_type.meta.start.line,
                )

            all_members[member_name] = self.parse_type(member_type)

        # As of Python 3.7, dict preserves insertion order.
        return cg.UnresolvedStructType(tuple((n, t) for n, t in all_members.items()))

    @classmethod
    def parse_and_resolve(
        cls,
        program: cg.Program,
        node: lp.Type | lp.CompileTimeConstant,
        generic_mapping: cg.GenericMapping,
    ) -> cg.SpecializationItem:
        unresolved_mapping: UnresolvedGenericMapping = {}
        for key, item in generic_mapping.mapping.items():
            if isinstance(item, cg.Type):
                unresolved_mapping[key.name] = cg.UnresolvedTypeWrapper(item)
            else:
                assert isinstance(item, int)
                unresolved_mapping[key.name] = cg.NumericLiteralConstant(item)

        result = cls(unresolved_mapping).parse_specialization(node)
        return program.symbol_table.resolve_specialization_item(result)


class TypeDefinitionsParser(lp.Interpreter):
    def __init__(
        self, file: str, include_hierarchy: list[str], program: cg.Program
    ) -> None:
        super().__init__()

        self._program = program
        self._file = file
        self._include_hierarchy = tuple(include_hierarchy)

    def parse_file(self, lines: list[lp.TopLevelFeature]) -> None:
        for line in lines:
            if isinstance(line, lp.Typedef):
                self.parse(line, None)

    def Typedef(self, node: lp.Typedef) -> None:
        loc = SourceLocation(node.meta.start.line, self._file, self._include_hierarchy)

        generic_mapping: UnresolvedGenericMapping = {}
        generic_definitions: list[cg.GenericArgument] = []

        for generic_definition in node.generic_definitions:
            definition = parse_generic_definition(generic_definition)
            generic_definitions.append(definition)

            if definition.name in generic_mapping:
                raise RepeatedGenericName(
                    generic_definition.name,
                    generic_definition.meta.start.line,
                    node.name,
                )

            generic_mapping[definition.name] = (
                cg.GenericValueReference(definition.name)
                if definition.is_value_arg
                else cg.UnresolvedGenericType(definition.name)
            )

        parser = TypeParser(generic_mapping)
        specialization = [
            parser.parse_specialization(item) for item in node.specialization
        ]
        rhs_type = parser.parse_type(node.alias)

        self._program.symbol_table.add_type(
            cg.Typedef.construct(
                node.name,
                tuple(generic_definitions),
                tuple(specialization),
                rhs_type,
                loc,
            )
        )


class FunctionSignatureParser(lp.Interpreter):
    def __init__(self, program: cg.Program) -> None:
        super().__init__()

        self._current_file: str | None = None
        self._current_di_file: cg.DIFile | None = None
        self._include_hierarchy: tuple[str, ...] | None = None

        self._program = program
        self._function_mapping: dict[UUID, lp.GenericFunctionDefinition] = {}

    def parse_file(
        self,
        file_name: str,
        include_hierarchy: list[str],
        top_level_features: list[lp.TopLevelFeature],
        di_file: cg.DIFile,
    ) -> None:
        self._current_file = file_name
        self._current_di_file = di_file
        self._include_hierarchy = tuple(include_hierarchy)
        for feature in top_level_features:
            if isinstance(feature, lp.FunctionDefinition):
                self.parse(feature, None)

        self._current_file = None
        self._include_hierarchy = None

    def get_functions_to_codegen(
        self,
    ) -> Generator[
        tuple[
            cg.FunctionDeclaration,
            cg.FunctionSignature,
            lp.GenericFunctionDefinition | None,
        ],
        None,
        None,
    ]:
        while to_generate := self._program.symbol_table.get_next_function_to_codegen():
            declaration, signature = to_generate

            body = None
            if not declaration.is_foreign:
                body = self._function_mapping[declaration.uuid]

            yield declaration, signature, body

    def GenericFunctionDefinition(self, node: lp.GenericFunctionDefinition) -> None:
        assert self._current_di_file is not None
        assert self._current_file is not None
        assert self._include_hierarchy is not None

        fn_name = (
            f"__builtin_{node.name}"
            if isinstance(node, lp.ImplicitFunction)
            else node.name
        )

        location = SourceLocation(
            node.meta.start.line,
            self._current_file,
            self._include_hierarchy,
        )

        generic_mapping: UnresolvedGenericMapping = {}
        generic_definitions: list[cg.GenericArgument] = []

        # One for the type and one for the actual args.
        variadic_type_pack_name: str | None = None
        variadic_args_pack_name: str | None = None

        for generic in node.generic_definitions:
            if generic.is_packed:
                # NOTE enforced by the grammar. We might want to relax this
                # later and allow more than one pack per function.
                assert variadic_type_pack_name is None
                variadic_type_pack_name = generic.name
                continue

            generic_definitions.append(parse_generic_definition(generic))
            if isinstance(generic, lp.TypeGenericDefinition):
                generic_mapping[generic.name] = cg.UnresolvedGenericType(generic.name)
            else:
                assert isinstance(generic, lp.NumericGenericDefinition)
                generic_mapping[generic.name] = cg.GenericValueReference(generic.name)

        parser = TypeParser(generic_mapping)
        specialization = [
            parser.parse_specialization(item) for item in node.specialization
        ]
        unresolved_return = parser.parse_type(node.return_)

        # TODO: this is a temporary hack
        if len(node.args) > 0 and isinstance(node.args[-1][1], lp.PackType):
            variadic_args_pack_name, packed_type = node.args.pop()

            # For now a packed type must be a plain type name
            assert isinstance(packed_type, lp.PackType)
            assert isinstance(packed_type.type_, lp.NamedType)
            assert len(packed_type.type_.specialization) == 0

        arg_names = [arg[0] for arg in node.args]
        arg_types = [parser.parse_type(arg[1]) for arg in node.args]

        fn_declaration = cg.FunctionDeclaration.construct(
            fn_name,
            False,
            tuple(generic_definitions),
            tuple(specialization),
            tuple(arg_names),
            variadic_args_pack_name,
            tuple(arg_types),
            variadic_type_pack_name,
            unresolved_return,
            location,
            node.meta,
            self._current_di_file,
        )
        self._function_mapping[fn_declaration.uuid] = node
        self._program.symbol_table.add_function(fn_declaration)

    def ForeignFunction(self, node: lp.ForeignFunction) -> None:
        assert self._current_di_file is not None
        assert self._current_file is not None
        assert self._include_hierarchy is not None

        parser = TypeParser({})
        fn_declaration = cg.FunctionDeclaration.construct(
            node.name,
            True,
            tuple(),
            tuple(),
            tuple(arg[0] for arg in node.args),
            None,
            tuple(parser.parse_type(arg[1]) for arg in node.args),
            None,
            parser.parse_type(node.return_),
            SourceLocation(
                node.meta.start.line, self._current_file, self._include_hierarchy
            ),
            node.meta,
            self._current_di_file,
        )
        self._program.symbol_table.add_function(fn_declaration)


class ImportParser(lp.Interpreter):
    def __init__(
        self,
        program: cg.Program,
        fn_parser: FunctionSignatureParser,
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

    def parse_file(self, lines: list[lp.TopLevelFeature]) -> None:
        for line in lines:
            if isinstance(line, lp.RequireOnce):
                self.parse(line, None)

    def RequireOnce(self, node: lp.RequireOnce) -> None:
        try:
            self._require_once_impl(node.path)
        except GrapheneError as exc:
            raise ErrorWithLineInfo(
                exc.message,
                node.meta.start.line,
                f'@require_once "{node.path}"',
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
        return self.expression().result_type


def is_flattened_expression_iterable(
    exprs: Iterable[Any],
) -> TypeGuard[Iterable[FlattenedExpression]]:
    # https://github.com/python/mypy/issues/3497#issuecomment-1083747764
    return all(isinstance(expr, FlattenedExpression) for expr in exprs)


class ExpressionParser(lp.Interpreter):
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

    def parse_expr(self, expr: lp.Expression) -> FlattenedExpression:
        return self.parse(expr, FlattenedExpression)

    def HexConstant(self, node: lp.HexConstant) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.UIntType(), node.value, node.meta)
        return FlattenedExpression([const_expr])

    def GenericIdentifierConstant(
        self, node: lp.GenericIdentifierConstant
    ) -> FlattenedExpression:
        argument = cg.GenericArgument(node.value, True)
        if argument not in self._generic_mapping.mapping:
            raise FailedLookupError("numeric generic", f"[{node.value}, ...]")

        mapped_value = self._generic_mapping.mapping[argument]
        assert isinstance(mapped_value, int)  # TODO: user facing error
        const_expr = cg.ConstantExpression(cg.IntType(), str(mapped_value), node.meta)
        return FlattenedExpression([const_expr])

    def FloatConstant(self, node: lp.FloatConstant) -> FlattenedExpression:
        const_expr = cg.ConstantExpression(cg.IEEEFloat(32), node.value, node.meta)
        return FlattenedExpression([const_expr])

    def IntConstant(self, node: lp.IntConstant) -> FlattenedExpression:
        # TODO: the parser has already decoded this, why are we undoing it?
        const_expr = cg.ConstantExpression(cg.IntType(), str(node.value), node.meta)
        return FlattenedExpression([const_expr])

    def BoolConstant(self, node: lp.BoolConstant) -> FlattenedExpression:
        # TODO: the parser has already decoded this, why are we undoing it?
        value = "true" if node.value else "false"
        const_expr = cg.ConstantExpression(cg.BoolType(), value, node.meta)
        return FlattenedExpression([const_expr])

    def StringConstant(self, node: lp.StringConstant) -> FlattenedExpression:
        str_static_storage = self._program.add_static_string(node.value, node.meta)
        expr = FlattenedExpression(
            [cg.VariableReference(str_static_storage, node.meta)]
        )

        # Implicitly take reference to string literal
        return expr.add_parent(
            cg.BorrowExpression(expr.expression(), cg.Type.Kind.CONST_REF, node.meta)
        )

    def OperatorUse(self, node: lp.OperatorUse) -> FlattenedExpression:
        lhs = self.parse_expr(node.lhs)
        rhs = self.parse_expr(node.rhs)

        flattened_expr = FlattenedExpression([])
        flattened_expr.subexpressions.extend(lhs.subexpressions)
        flattened_expr.subexpressions.extend(rhs.subexpressions)

        call_expr = self._program.lookup_call_expression(
            node.name,
            [],  # Don't specialize operators
            [lhs.expression(), rhs.expression()],
            node.meta,
        )
        return flattened_expr.add_parent(call_expr)

    def UnaryOperatorUse(self, node: lp.UnaryOperatorUse) -> FlattenedExpression:
        rhs = self.parse_expr(node.rhs)

        flattened_expr = FlattenedExpression(rhs.subexpressions)
        call_expr = self._program.lookup_call_expression(
            node.name,
            [],  # Don't specialize operators
            [rhs.expression()],
            node.meta,
        )
        return flattened_expr.add_parent(call_expr)

    def LogicalOperatorUse(self, node: lp.LogicalOperatorUse) -> FlattenedExpression:
        lhs = self.parse_expr(node.lhs)
        rhs = self.parse_expr(node.rhs)

        flattened_expr = FlattenedExpression([])
        flattened_expr.subexpressions.extend(lhs.subexpressions)
        return flattened_expr.add_parent(
            cg.LogicalOperator(
                node.name,
                self._function.get_next_label_id(),
                lhs.expression(),
                rhs.subexpressions[:-1],
                rhs.expression(),
                node.meta,
            )
        )

    def _function_call_inner(
        self,
        fn_name: str,
        fn_specialization: Iterable[lp.CompileTimeConstant | lp.Type],
        fn_args: Iterable[FlattenedExpression],
        meta: lp.Meta,
    ) -> FlattenedExpression:
        fn_call_args = []
        flattened_expr = FlattenedExpression([])
        for arg in fn_args:
            fn_call_args.append(arg.expression())
            flattened_expr.subexpressions.extend(arg.subexpressions)

        specialization = [
            TypeParser.parse_and_resolve(
                self._program,
                item,
                self._generic_mapping,
            )
            for item in fn_specialization
        ]

        call_expr = self._program.lookup_call_expression(
            fn_name, specialization, fn_call_args, meta
        )
        return flattened_expr.add_parent(call_expr)

    @staticmethod
    def _extract_parameter_pack(
        args: list[lp.Expression],
    ) -> tuple[list[lp.Expression], str | None]:
        if len(args) == 0:
            return args, None

        if isinstance(args[-1], lp.PackExpansion):
            # TODO: support smarter expansions
            expansion = args[-1]
            assert isinstance(expansion.expression, lp.VariableAccess)
            return args[:-1], expansion.expression.name

        return args, None

    def _function_call_outer(
        self,
        fn_name: str,
        fn_specialization: Iterable[lp.CompileTimeConstant | lp.Type],
        fn_args: Iterable[lp.Expression],
        ufcs_expression: lp.Expression | None,
        meta: lp.Meta,
    ) -> FlattenedExpression:
        normal_args, pack_name = self._extract_parameter_pack(list(fn_args))
        unresolved_args = [self.parse_expr(arg) for arg in normal_args]

        if pack_name:
            variadic_vars = self._scope.search_for_generic_pack(pack_name)
            unresolved_args.extend(
                FlattenedExpression([cg.VariableReference(var, meta)])
                for var in variadic_vars
            )

        # TODO perhaps we shouldn't always borrow this, although this is a bit
        # tricky as we haven't done overload resolution yet (which depends on
        # whether we borrow or not). A solution would be to borrow if we can,
        # otherwise pass an unborrowed/const-reference and let overload
        # resolution figure it out, although this isn't very explicit.
        if ufcs_expression is not None:
            this_arg = self.parse_expr(ufcs_expression)
            # TODO: should there be syntax for this? It seems a bit adhoc
            match this_arg.expression().underlying_indirection_kind:
                case cg.Type.Kind.CONST_REF:
                    this_arg.add_parent(
                        cg.BorrowExpression(
                            this_arg.expression(), cg.Type.Kind.CONST_REF, meta
                        )
                    )
                case cg.Type.Kind.MUTABLE_REF:
                    this_arg.add_parent(
                        cg.BorrowExpression(
                            this_arg.expression(),
                            cg.Type.Kind.MUTABLE_OR_CONST_REF,
                            meta,
                        )
                    )
                case cg.Type.Kind.VALUE:
                    # (&a):fn(), (&mut a):fn(), or rvalue():fn();
                    # We do not allow the 3rd option
                    possible = this_arg.expression().result_type_as_if_borrowed
                    if possible.storage_kind == cg.Type.Kind.VALUE:
                        raise BorrowWithNoAddressError(
                            possible.format_for_output_to_user()
                        )
                case _:
                    # cg.Type.Kind.MUTABLE_OR_CONST_REF shouldn't be possible.
                    raise AssertionError

            unresolved_args.insert(0, this_arg)

        return self._function_call_inner(
            fn_name, fn_specialization, unresolved_args, meta
        )

    def FunctionCall(self, node: lp.FunctionCall) -> FlattenedExpression:
        return self._function_call_outer(
            node.name, node.specialization, node.args, None, node.meta
        )

    def UFCS_Call(self, node: lp.UFCS_Call) -> FlattenedExpression:
        return self._function_call_outer(
            node.name, node.specialization, node.args, node.expression, node.meta
        )

    def IndexOperator(self, node: lp.IndexOperator) -> FlattenedExpression:
        return self._function_call_outer(
            "[]", [], node.indexes, node.expression, node.meta
        )

    def VariableAccess(self, node: lp.VariableAccess) -> FlattenedExpression:
        var = self._scope.search_for_variable(node.name)
        if var is None:
            raise FailedLookupError("variable", node.name)

        return FlattenedExpression([cg.VariableReference(var, node.meta)])

    def _ensure_pointer_is_available(
        self, expr: FlattenedExpression, meta: lp.Meta
    ) -> FlattenedExpression:
        # Copy expression to stack if it is not a pointer
        if expr.expression().has_address:
            return expr

        temp_var = cg.StackVariable(
            "", expr.type(), True, True, meta, self._function._di_file
        )  # FIXME
        self._scope.add_variable(temp_var)

        expr.subexpressions.append(
            cg.VariableInitialize(temp_var, expr.expression(), meta)
        )
        return expr.add_parent(cg.VariableReference(temp_var, meta))

    def StructIndexAccess(self, node: lp.StructIndexAccess) -> FlattenedExpression:
        lhs = self.parse_expr(node.expression)
        struct_access = cg.StructMemberAccess(lhs.expression(), node.member, node.meta)
        return lhs.add_parent(struct_access)

    def Borrow(self, node: lp.Borrow) -> FlattenedExpression:
        lhs = self.parse_expr(node.expression)
        return lhs.add_parent(
            cg.BorrowExpression(
                lhs.expression(),
                cg.Type.Kind.MUTABLE_REF if node.is_mut else cg.Type.Kind.CONST_REF,
                node.meta,
            )
        )

    def UnnamedInitializerList(
        self, node: lp.UnnamedInitializerList
    ) -> FlattenedExpression:
        arg_exprs: list[cg.TypedExpression] = []
        combined_flattened = FlattenedExpression([])
        for arg in node.args:
            arg_expr = self.parse_expr(arg)
            combined_flattened.subexpressions.extend(arg_expr.subexpressions)
            arg_exprs.append(arg_expr.expression())

        return combined_flattened.add_parent(
            cg.UnnamedInitializerList(arg_exprs, node.meta)
        )

    def NamedInitializerList(
        self, node: lp.NamedInitializerList
    ) -> FlattenedExpression:
        names: list[str] = []
        member_exprs: list[cg.TypedExpression] = []
        combined_flattened = FlattenedExpression([])
        for name, expression in node.args:
            names.append(name)
            expr = self.parse_expr(expression)
            combined_flattened.subexpressions.extend(expr.subexpressions)
            member_exprs.append(expr.expression())

        return combined_flattened.add_parent(
            cg.NamedInitializerList(member_exprs, names, node.meta)
        )


def generate_standalone_expression(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.Expression,
    generic_mapping: cg.GenericMapping,
) -> None:
    parser = ExpressionParser(program, function, scope, generic_mapping)
    flattened_expr = parser.parse_expr(node)
    scope.add_generatable(flattened_expr.subexpressions)


def generate_return_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.Return,
    generic_mapping: cg.GenericMapping,
) -> None:
    return_type = function.get_signature().return_type

    if node.expression is None:
        scope.add_generatable(cg.ReturnStatement(return_type, node.meta))
        return

    parser = ExpressionParser(program, function, scope, generic_mapping)
    flattened_expr = parser.parse_expr(node.expression)
    scope.add_generatable(flattened_expr.subexpressions)

    expr = cg.ReturnStatement(
        return_type, node.meta, returned_expr=flattened_expr.expression()
    )
    scope.add_generatable(expr)


def generate_variable_declaration(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.VariableDeclaration,
    generic_mapping: cg.GenericMapping,
) -> None:
    rhs = None
    if node.expression is not None:
        parser = ExpressionParser(program, function, scope, generic_mapping)
        rhs = parser.parse_expr(node.expression)

    var_type = TypeParser.parse_and_resolve(program, node.type_, generic_mapping)
    assert isinstance(var_type, cg.Type)
    if var_type.definition.is_void:
        raise VoidVariableDeclaration(
            "variable", node.variable, var_type.format_for_output_to_user()
        )

    var = cg.StackVariable(
        node.variable,
        var_type,
        node.is_mut,
        rhs is not None,
        node.meta,
        function._di_file,
    )
    scope.add_variable(var)

    if rhs is None:
        return

    scope.add_generatable(rhs.subexpressions)
    scope.add_generatable(cg.VariableInitialize(var, rhs.expression(), node.meta))


def generate_if_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.If,
    generic_mapping: cg.GenericMapping,
) -> None:
    parser = ExpressionParser(program, function, scope, generic_mapping)
    condition_expr = parser.parse_expr(node.condition)

    scope.add_generatable(condition_expr.subexpressions)

    if_scope = cg.Scope(function.get_next_scope_id(), node.meta, scope)
    generate_body(program, function, if_scope, node.if_scope, generic_mapping)

    # Note: this looks like a redundant scope when the else branch is empty but I've
    #       chosen to explicitly codegen it here so we can generate destructors in
    #       the else branch (eg. if it was moved in the if)
    else_scope = cg.Scope(function.get_next_scope_id(), node.else_scope.meta, scope)
    generate_body(program, function, else_scope, node.else_scope, generic_mapping)

    if_statement = cg.IfElseStatement(
        condition_expr.expression(), if_scope, else_scope, node.meta
    )
    scope.add_generatable(if_statement)


def generate_while_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.While,
    generic_mapping: cg.GenericMapping,
) -> None:
    parser = ExpressionParser(program, function, scope, generic_mapping)
    condition_expr = parser.parse_expr(node.condition)

    while_scope_id = function.get_next_scope_id()

    inner_scope = cg.Scope(
        function.get_next_scope_id(), node.scope.meta, scope, is_inside_loop=True
    )
    generate_body(program, function, inner_scope, node.scope, generic_mapping)

    scope.add_generatable(
        cg.WhileStatement(
            while_scope_id,
            condition_expr.expression(),
            condition_expr.subexpressions,
            inner_scope,
            node.meta,
        )
    )


def generate_for_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.For,
    generic_mapping: cg.GenericMapping,
) -> None:
    # Outer scope
    outer_scope = cg.Scope(function.get_next_scope_id(), node.meta, outer_scope=scope)

    #    Produce the iterator
    parser = ExpressionParser(program, function, scope, generic_mapping)
    iter_expr = parser.parse_expr(node.iterator)

    #    Save the iterator onto the stack (for referencing)
    store_in_mutable = not iter_expr.type().storage_kind.is_reference()
    iter_variable = cg.StackVariable(
        f"__for_iter_{outer_scope.id}",
        iter_expr.type(),
        store_in_mutable,
        True,
        node.meta,
        function._di_file,
    )
    outer_scope.add_variable(iter_variable)
    outer_scope.add_generatable(iter_expr.subexpressions)
    outer_scope.add_generatable(
        cg.VariableInitialize(iter_variable, iter_expr.expression(), node.meta)
    )
    var_ref = cg.VariableReference(iter_variable, node.meta)
    borrowed_iter_expr = cg.BorrowExpression(
        var_ref, cg.Type.Kind.MUTABLE_REF, node.meta
    )
    outer_scope.add_generatable([var_ref, borrowed_iter_expr])

    # Inner scope
    inner_scope = cg.Scope(
        function.get_next_scope_id(), node.scope.meta, outer_scope, is_inside_loop=True
    )

    has_next_expr = program.lookup_call_expression(
        "__builtin_has_next", [], [borrowed_iter_expr], node.scope.meta
    )
    get_next_expr = program.lookup_call_expression(
        "__builtin_get_next", [], [borrowed_iter_expr], node.scope.meta
    )
    inner_scope.add_generatable(get_next_expr)

    iter_result_variable = cg.StackVariable(
        node.variable,
        get_next_expr.result_type,
        False,
        True,
        node.meta,
        function._di_file,
    )
    inner_scope.add_variable(iter_result_variable)
    inner_scope.add_generatable(
        cg.VariableInitialize(iter_result_variable, get_next_expr, node.scope.meta)
    )

    # For loop is just syntax sugar for a while loop
    generate_body(program, function, inner_scope, node.scope, generic_mapping)

    outer_scope.add_generatable(
        cg.WhileStatement(
            inner_scope.id, has_next_expr, [has_next_expr], inner_scope, node.meta
        )
    )
    scope.add_generatable(outer_scope)


def generate_continue_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.Return,
    generic_mapping: cg.GenericMapping,
) -> None:
    if not scope.is_inside_loop():
        raise NotInLoopStatementError("continue")

    scope.add_generatable(cg.ContinueStatement(node.meta))


def generate_break_statement(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.Return,
    generic_mapping: cg.GenericMapping,
) -> None:
    if not scope.is_inside_loop():
        raise NotInLoopStatementError("break")

    scope.add_generatable(cg.BreakStatement(node.meta))


def generate_assignment(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    node: lp.Assignment,
    generic_mapping: cg.GenericMapping,
) -> None:
    parser = ExpressionParser(program, function, scope, generic_mapping)
    lhs = parser.parse_expr(node.lhs)
    rhs = parser.parse_expr(node.rhs)

    scope.add_generatable(lhs.subexpressions)
    scope.add_generatable(rhs.subexpressions)

    if node.operator == "=":
        scope.add_generatable(
            cg.Assignment(lhs.expression(), rhs.expression(), node.meta)
        )
    else:
        borrowed_lhs = cg.BorrowExpression(
            lhs.expression(), cg.Type.Kind.MUTABLE_REF, node.meta
        )
        scope.add_generatable(borrowed_lhs)
        scope.add_generatable(
            program.lookup_call_expression(
                node.operator, [], [borrowed_lhs, rhs.expression()], node.meta
            )
        )


def generate_scope_body(
    program: cg.Program,
    function: cg.Function,
    outer_scope: cg.Scope,
    node: lp.Scope,
    generic_mapping: cg.GenericMapping,
) -> None:
    inner_scope = cg.Scope(function.get_next_scope_id(), node.meta, outer_scope)
    generate_body(program, function, inner_scope, node, generic_mapping)
    outer_scope.add_generatable(inner_scope)


def generate_body(
    program: cg.Program,
    function: cg.Function,
    scope: cg.Scope,
    body: lp.Scope,
    generic_mapping: cg.GenericMapping,
) -> None:
    generators = {
        "Assignment": generate_assignment,
        "Break": generate_break_statement,
        "Continue": generate_continue_statement,
        "Expression": generate_standalone_expression,
        "For": generate_for_statement,
        "If": generate_if_statement,
        "Return": generate_return_statement,
        "Scope": generate_scope_body,
        "VariableDeclaration": generate_variable_declaration,
        "While": generate_while_statement,
    }

    for line in body.lines:
        try:
            for candidate in type(line).mro():
                name = candidate.__name__
                if name in generators:
                    break
            else:
                raise AssertionError

            generators[name](program, function, scope, line, generic_mapping)
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
    body: lp.GenericFunctionDefinition | None,
) -> None:
    try:
        fn = cg.Function(
            declaration.arg_names,
            signature,
            declaration.pack_type_name,
            declaration.di_file,
            program.di_compile_unit,
            declaration.meta,
        )
    except GrapheneError as err:
        assert isinstance(declaration.loc, SourceLocation)
        raise ErrorWithLineInfo(
            err.message, declaration.loc.line, "function declaration"
        ) from err

    if body is None:
        assert declaration.is_foreign
        return

    generate_body(program, fn, fn.top_level_scope, body.scope, declaration.mapping)

    # We cannot omit the "ret" instruction from LLVM IR. If the function returns
    # void, then we can add it ourselves, otherwise the user needs to fix it.
    if not fn.top_level_scope.is_return_guaranteed():
        if fn.get_signature().return_type.definition.is_void:
            # FIXME the meta here should come from the last } of the function.
            fn.top_level_scope.add_generatable(
                cg.ReturnStatement(cg.VoidType(), declaration.meta)
            )
        else:
            raise MissingFunctionReturn(
                fn.get_signature().user_facing_name,
                body.meta.end.line,
            )

    program.add_function_body(fn)


def append_file_to_program(
    program: cg.Program,
    function_parser: FunctionSignatureParser,
    file_path: ResolvedPath,
    include_path: list[Path],
    included_from: list[ResolvedPath],
    already_processed: set[ResolvedPath],
    di_file: cg.DIFile | None = None,
) -> None:
    already_processed.add(file_path)
    if di_file is None:
        di_file = program.add_secondary_file(Path(file_path))

    try:
        top_level_features = lp.run_lexer_parser(Path(file_path))

        ImportParser(
            program,
            function_parser,
            [*include_path, Path(file_path).parent],
            [*included_from, file_path],
            already_processed,
        ).parse_file(top_level_features)

        TypeDefinitionsParser(
            str(file_path), list(map(str, included_from)), program
        ).parse_file(top_level_features)

        function_parser.parse_file(
            str(file_path), list(map(str, included_from)), top_level_features, di_file
        )
        program.symbol_table.resolve_all_non_generics()
    except ErrorWithLineInfo as exc:
        raise ErrorWithLocationInfo(
            exc.message,
            SourceLocation(exc.line, str(file_path), tuple(map(str, included_from))),
            exc.context,
        ) from exc


def generate_ir_from_source(
    file_path: Path, include_path: list[Path], *, debug_compiler: bool = False
) -> str:
    program = cg.Program(file_path)
    try:
        # Initial pass resolves all builtin types
        program.symbol_table.resolve_all_non_generics()

        fn_parser = FunctionSignatureParser(program)
        append_file_to_program(
            program,
            fn_parser,
            ResolvedPath(file_path),
            include_path,
            [],
            set(),
            program.di_files[0],
        )

        for declaration, signature, body in fn_parser.get_functions_to_codegen():
            try:
                generate_function(program, declaration, signature, body)
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

    return program.generate_ir()

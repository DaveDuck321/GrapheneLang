from collections.abc import Iterator
from functools import cached_property
from itertools import count
from pathlib import Path

from glang import target
from glang.codegen.builtin_callables import get_builtin_callables
from glang.codegen.builtin_types import (
    AnonymousType,
    CharArrayDefinition,
    FunctionSignature,
    IntType,
    get_builtin_types,
)
from glang.codegen.debug import (
    DICompileUnit,
    DIFile,
    DISubprogram,
    DISubroutineType,
    ModuleFlags,
    ModuleFlagsBehavior,
    Tag,
)
from glang.codegen.expressions import FunctionCallExpression, FunctionParameter
from glang.codegen.generatable import (
    Scope,
    StackVariable,
    StaticVariable,
    VariableInitialize,
)
from glang.codegen.interfaces import (
    IRContext,
    IROutput,
    SpecializationItem,
    Type,
    TypedExpression,
)
from glang.codegen.strings import encode_string
from glang.codegen.type_conversions import get_implicit_conversion_cost
from glang.codegen.type_resolution import FunctionDeclaration, SymbolTable
from glang.codegen.user_facing_errors import (
    InvalidMainReturnType,
    VoidVariableDeclaration,
)
from glang.parser.lexer_parser import Meta
from glang.utils.stack import Stack


class Function:
    def __init__(
        self,
        parameter_names: tuple[str, ...],
        signature: FunctionSignature,
        parameter_pack_name: str | None,
        di_file: DIFile,
        di_unit: DICompileUnit,
        meta: Meta,
    ) -> None:
        if parameter_pack_name is None:
            assert len(parameter_names) == len(signature.arguments)

        self._signature = signature
        self._di_file = di_file
        self._di_unit = di_unit
        self._meta = meta

        # These two counters could be merged.
        self.label_id_iter = count()
        self.scope_id_iter = count()
        self.top_level_scope = Scope(self.get_next_scope_id(), self._meta)

        # Implicit stack variable allocation for parameters
        #   TODO: constant parameters (when/ if added to grammar)
        self._parameters: list[FunctionParameter] = []
        for param_name, param_type in zip(
            parameter_names, signature.arguments, strict=False
        ):
            self._add_parameter(param_name, param_type)

        if parameter_pack_name is not None:
            packed_args = signature.arguments[len(parameter_names) :]
            self.top_level_scope.add_generic_pack(parameter_pack_name, len(packed_args))

            for i, param_type in enumerate(packed_args):
                self._add_parameter(f"{parameter_pack_name}{i}", param_type)

        # Check the signature
        if signature.is_main:
            if get_implicit_conversion_cost(signature.return_type, IntType()) is None:
                raise InvalidMainReturnType(
                    signature.return_type.format_for_output_to_user(True)
                )

        for arg_name, arg in zip(
            parameter_names, self._signature.arguments, strict=False
        ):
            if arg.definition.is_void:
                raise VoidVariableDeclaration(
                    "argument", arg_name, arg.format_for_output_to_user(True)
                )

    def _add_parameter(self, param_name: str, param_type: Type):
        # FIXME pass the true meta.
        fn_param = FunctionParameter(param_type, self._meta)
        self._parameters.append(fn_param)
        self.top_level_scope.add_generatable(fn_param)

        # Allow assignments to parameters passed by value (but our references
        # still need to be immutable).
        is_value_type = param_type.storage_kind == Type.Kind.VALUE

        # FIXME pass the true meta.
        fn_param_var = StackVariable(
            param_name, param_type, is_value_type, True, self._meta, self._di_file
        )
        self.top_level_scope.add_variable(fn_param_var)

        # FIXME pass the true meta.
        fn_param_var_assignment = VariableInitialize(fn_param_var, fn_param, self._meta)
        self.top_level_scope.add_generatable(fn_param_var_assignment)

    def __repr__(self) -> str:
        return f"Function({self._signature!r})"

    @cached_property
    def mangled_name(self) -> str:
        return self._signature.mangled_name

    def get_signature(self) -> FunctionSignature:
        return self._signature

    @property
    def is_foreign(self) -> bool:
        return self._signature.is_foreign

    def get_next_scope_id(self) -> int:
        return next(self.scope_id_iter)

    def get_next_label_id(self) -> int:
        return next(self.label_id_iter)

    def generate_ir(self, metadata_gen: Iterator[int]) -> IROutput:
        # https://llvm.org/docs/LangRef.html#functions
        assert not self.is_foreign
        if not self._signature.return_type.definition.is_void:
            assert self.top_level_scope.is_return_guaranteed()

        di_subroutine_type = DISubroutineType(
            next(metadata_gen), None, 0, Tag.unspecified_type
        )

        di_subprogram = DISubprogram(
            next(metadata_gen),
            self._signature.name,
            self._signature.mangled_name,
            di_subroutine_type,
            self._di_file,
            self._meta.start.line,
            self._di_unit,
            not self._signature.is_foreign,
        )

        ctx = IRContext(count(0), metadata_gen, di_subprogram, Stack())

        for param in self._parameters:
            param.set_reg(ctx.next_reg())

        args_ir = ", ".join(
            param.ir_ref_with_type_annotation for param in self._parameters
        )

        def indent_ir(lines: list[str]):
            return [line if line.endswith(":") else f"  {line}" for line in lines]

        body_ir = self.top_level_scope.generate_ir(ctx)

        return IROutput(
            [
                (
                    f"define dso_local {self._signature.return_type.ir_type}"
                    f" @{self.mangled_name}({args_ir}) !dbg !{di_subprogram.id} {{"
                ),
                "begin:",  # Name the implicit basic block
                *indent_ir(body_ir.lines),
                "}",
            ],
            {di_subroutine_type, di_subprogram, *body_ir.metadata},
        )


class Program:
    def __init__(self, file: Path) -> None:
        super().__init__()
        self._builtin_callables = get_builtin_callables()

        self._fn_bodies: list[Function] = []
        self.symbol_table = SymbolTable()

        self._string_cache: dict[str, StaticVariable] = {}
        self._static_variables: list[StaticVariable] = []

        self._has_main: bool = False

        self._functions_to_codegen: list[tuple[FunctionSignature, FunctionDeclaration]]

        self._metadata_gen = count(0)

        self.di_files = [DIFile(next(self._metadata_gen), file)]
        self.di_compile_unit = DICompileUnit(next(self._metadata_gen), self.di_files[0])

        for builtin_type in get_builtin_types():
            self.symbol_table.add_builtin_type(builtin_type)

    def lookup_call_expression(
        self,
        fn_name: str,
        fn_specialization: list[SpecializationItem],
        fn_args: list[TypedExpression],
        meta: Meta,
    ) -> TypedExpression:
        if fn_name in self._builtin_callables:
            return self._builtin_callables[fn_name](fn_specialization, fn_args, meta)

        signature, _ = self.symbol_table.lookup_function(
            fn_name, fn_specialization, fn_args, evaluated_context=True
        )
        return FunctionCallExpression(signature, fn_args, meta)

    def add_static_string(self, string: str, meta: Meta) -> StaticVariable:
        if string in self._string_cache:
            return self._string_cache[string]

        encoded_str, encoded_length = encode_string(string)
        str_type = AnonymousType(CharArrayDefinition(encoded_str, encoded_length))
        static_storage = StaticVariable(
            f'string literal: "{string}"',
            str_type,
            False,
            encoded_str,
            meta,
            self.di_files[0],  # FIXME this is wrong.
        )
        self.add_static_variable(static_storage)

        self._string_cache[string] = static_storage
        return static_storage

    def add_static_variable(self, var: StaticVariable) -> None:
        self._static_variables.append(var)

    def add_function_body(self, function: Function) -> None:
        self._fn_bodies.append(function)

    def add_secondary_file(self, path: Path) -> DIFile:
        di_file = DIFile(next(self._metadata_gen), path)
        self.di_files.append(di_file)
        return di_file

    def generate_ir(self) -> str:
        output = IROutput(metadata={self.di_compile_unit, *self.di_files})

        output.lines.append(f'target datalayout = "{target.get_datalayout()}"')
        output.lines.append(f'target triple = "{target.get_target_triple()}"')

        output.lines.append("")
        # FIXME the file passed in is wrong (although we aren't using it yet).
        ctx = IRContext(count(0), count(0), self.di_files[0], Stack())
        for variable in self._static_variables:
            output.extend(variable.generate_ir(ctx))

        output.lines.append("")
        output.lines.extend(self.symbol_table.get_ir_for_initialization())

        # We need to declare any debugger intrinsics we use for some reason.
        output.lines.append("")
        output.lines.append(
            "declare void @llvm.dbg.declare(metadata, metadata, metadata)"
        )

        output.lines.append("")
        for func in self._fn_bodies:
            output.extend(func.generate_ir(self._metadata_gen))

        dwarf_version_metadata = ModuleFlags(
            next(self._metadata_gen), ModuleFlagsBehavior.Error, "Dwarf Version", 4
        )
        debug_info_version_metadata = ModuleFlags(
            next(self._metadata_gen), ModuleFlagsBehavior.Error, "Debug Info Version", 3
        )

        output.metadata.update((dwarf_version_metadata, debug_info_version_metadata))

        source = "\n".join(output.lines)
        source += "\n\n"

        source += "\n".join(
            (
                f"!llvm.dbg.cu = !{{!{self.di_compile_unit.id}}}",
                f"!llvm.module.flags = !{{!{dwarf_version_metadata.id}, !{debug_info_version_metadata.id}}}",
            )
        )
        source += "\n\n"

        source += "\n".join(f"!{m.id} = {m}" for m in output.metadata)

        return source

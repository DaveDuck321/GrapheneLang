from dataclasses import dataclass, field
from functools import cached_property
from itertools import count
from pathlib import Path
from typing import Iterator, Optional

from .. import target
from .builtin_callables import get_builtin_callables
from .builtin_types import (
    AnonymousType,
    CharArrayDefinition,
    FunctionSignature,
    IntType,
    get_builtin_types,
)
from .debug import (
    DICompileUnit,
    DIFile,
    DISubprogram,
    DISubroutineType,
    Metadata,
    MetadataFlag,
)
from .expressions import FunctionCallExpression, FunctionParameter
from .generatable import Scope, StackVariable, StaticVariable, VariableInitialize
from .interfaces import SpecializationItem, Type, TypedExpression
from .strings import encode_string
from .type_conversions import get_implicit_conversion_cost
from .type_resolution import FunctionDeclaration, SymbolTable
from .user_facing_errors import InvalidMainReturnType, VoidVariableDeclaration


@dataclass
class IROutput:
    lines: list[str] = field(default_factory=list)
    metadata: list[Metadata] = field(default_factory=list)

    def extend(self, other: "IROutput") -> None:
        self.lines.extend(other.lines)
        self.metadata.extend(other.metadata)


class Function:
    def __init__(
        self,
        parameter_names: tuple[str, ...],
        signature: FunctionSignature,
        parameter_pack_name: Optional[str],
        di_file: DIFile,
        di_unit: DICompileUnit,
    ) -> None:
        if parameter_pack_name is None:
            assert len(parameter_names) == len(signature.arguments)

        self._signature = signature
        self._di_file = di_file
        self._di_unit = di_unit

        # These two counters could be merged.
        self.label_id_iter = count()
        self.scope_id_iter = count()
        self.top_level_scope = Scope(self.get_next_scope_id())

        # Implicit stack variable allocation for parameters
        #   TODO: constant parameters (when/ if added to grammar)
        self._parameters: list[FunctionParameter] = []
        for param_name, param_type in zip(parameter_names, signature.arguments):
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

        for arg_name, arg in zip(parameter_names, self._signature.arguments):
            if arg.definition.is_void:
                raise VoidVariableDeclaration(
                    "argument", arg_name, arg.format_for_output_to_user(True)
                )

    def _add_parameter(self, param_name: str, param_type: Type):
        fn_param = FunctionParameter(param_type)
        self._parameters.append(fn_param)
        self.top_level_scope.add_generatable(fn_param)

        # Allow assignments to parameters passed by value (but our references
        # still need to be immutable).
        is_value_type = param_type.storage_kind == Type.Kind.VALUE

        fn_param_var = StackVariable(
            param_name, param_type, is_mutable=is_value_type, initialized=True
        )
        self.top_level_scope.add_variable(fn_param_var)

        fn_param_var_assignment = VariableInitialize(fn_param_var, fn_param)
        self.top_level_scope.add_generatable(fn_param_var_assignment)

    def __repr__(self) -> str:
        return f"Function({repr(self._signature)})"

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

        di_subroutine_type = DISubroutineType(next(metadata_gen))

        di_subprogram = DISubprogram(
            next(metadata_gen),
            self._signature.name,
            self._signature.mangled_name,
            di_subroutine_type,
            self._di_file,
            0,  # TODO line number.
            self._di_unit,
            not self._signature.is_foreign,
        )

        reg_gen = count(0)  # First register is %0

        for param in self._parameters:
            param.set_reg(next(reg_gen))

        args_ir = ", ".join(
            map(lambda param: param.ir_ref_with_type_annotation, self._parameters)
        )

        def indent_ir(lines: list[str]):
            return map(lambda line: line if line.endswith(":") else f"  {line}", lines)

        return IROutput(
            [
                (
                    f"define dso_local {self._signature.return_type.ir_type}"
                    f" @{self.mangled_name}({args_ir}) !dbg !{di_subprogram.id} {{"
                ),
                "begin:",  # Name the implicit basic block
                *indent_ir(self.top_level_scope.generate_ir(reg_gen)),
                "}",
            ],
            [di_subroutine_type, di_subprogram],
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

        self.di_file = DIFile(next(self._metadata_gen), file)
        self.di_compile_unit = DICompileUnit(next(self._metadata_gen), self.di_file)

        for builtin_type in get_builtin_types():
            self.symbol_table.add_builtin_type(builtin_type)

    def lookup_call_expression(
        self,
        fn_name: str,
        fn_specialization: list[SpecializationItem],
        fn_args: list[TypedExpression],
    ) -> TypedExpression:
        if fn_name in self._builtin_callables:
            return self._builtin_callables[fn_name](fn_specialization, fn_args)

        signature, _ = self.symbol_table.lookup_function(
            fn_name, fn_specialization, fn_args, True
        )
        return FunctionCallExpression(signature, fn_args)

    def add_static_string(self, string: str) -> StaticVariable:
        if string in self._string_cache:
            return self._string_cache[string]

        encoded_str, encoded_length = encode_string(string)
        str_type = AnonymousType(CharArrayDefinition(encoded_str, encoded_length))
        static_storage = StaticVariable(
            f'string literal: "{string}"', str_type, False, encoded_str
        )
        self.add_static_variable(static_storage)

        self._string_cache[string] = static_storage
        return static_storage

    def add_static_variable(self, var: StaticVariable) -> None:
        self._static_variables.append(var)

    def add_function_body(self, function: Function) -> None:
        self._fn_bodies.append(function)

    def generate_ir(self) -> str:
        output = IROutput([], [self.di_file, self.di_compile_unit])

        output.lines.append(f'target datalayout = "{target.get_datalayout()}"')
        output.lines.append(f'target triple = "{target.get_target_triple()}"')

        output.lines.append("")
        var_reg_gen = count(0)
        for variable in self._static_variables:
            output.lines.extend(variable.generate_ir(var_reg_gen))

        output.lines.append("")
        output.lines.extend(self.symbol_table.get_ir_for_initialization())

        output.lines.append("")
        for func in self._fn_bodies:
            output.extend(func.generate_ir(self._metadata_gen))

        # behaviour = 1 (Error)
        # Emits an error if two values disagree, otherwise the resulting value
        # is that of the operands.
        dwarf_version_metadata = MetadataFlag(
            next(self._metadata_gen), 1, "Dwarf Version", 4
        )
        debug_info_version_metadata = MetadataFlag(
            next(self._metadata_gen), 1, "Debug Info Version", 3
        )

        output.metadata.extend((dwarf_version_metadata, debug_info_version_metadata))

        source = "\n".join(output.lines)
        source += "\n\n"

        source += "\n".join(
            (
                f"!llvm.dbg.cu = !{{!{self.di_compile_unit.id}}}",
                f"!llvm.module.flags = !{{!{dwarf_version_metadata.id}, !{debug_info_version_metadata.id}}}",
            )
        )
        source += "\n\n"

        source += "\n".join(map(lambda m: f"!{m.id} = {m}", output.metadata))

        return source

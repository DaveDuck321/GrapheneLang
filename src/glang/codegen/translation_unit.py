from functools import cached_property
from itertools import count
from typing import Optional

from .. import target
from .builtin_callables import get_builtin_callables
from .builtin_types import (
    AnonymousType,
    CharArrayDefinition,
    FunctionSignature,
    IntType,
    get_builtin_types,
)
from .expressions import FunctionCallExpression, FunctionParameter
from .generatable import Scope, StackVariable, StaticVariable, VariableAssignment
from .interfaces import SpecializationItem, Type, TypedExpression
from .strings import encode_string
from .type_conversions import get_implicit_conversion_cost
from .type_resolution import SymbolTable
from .user_facing_errors import InvalidMainReturnType, VoidVariableDeclaration


class Function:
    def __init__(
        self,
        parameter_names: tuple[str, ...],
        signature: FunctionSignature,
        parameter_pack_name: Optional[str],
    ) -> None:
        if parameter_pack_name is None:
            assert len(parameter_names) == len(signature.arguments)

        self._signature = signature

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

        fn_param_var = StackVariable(param_name, param_type, False, True)
        self.top_level_scope.add_variable(fn_param_var)

        fn_param_var_assignment = VariableAssignment(fn_param_var, fn_param)
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

    def generate_ir(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#functions
        assert not self.is_foreign
        if not self._signature.return_type.definition.is_void:
            assert self.top_level_scope.is_return_guaranteed()

        reg_gen = count(0)  # First register is %0

        for param in self._parameters:
            param.set_reg(next(reg_gen))

        args_ir = ", ".join(
            map(lambda param: param.ir_ref_with_type_annotation, self._parameters)
        )

        def indent_ir(lines: list[str]):
            return map(lambda line: line if line.endswith(":") else f"  {line}", lines)

        return [
            (
                f"define dso_local {self._signature.return_type.ir_type}"
                f" @{self.mangled_name}({args_ir}) {{"
            ),
            "begin:",  # Name the implicit basic block
            *indent_ir(self.top_level_scope.generate_ir(reg_gen)),
            "}",
        ]


class Program:
    def __init__(self) -> None:
        super().__init__()
        self._builtin_callables = get_builtin_callables()

        self._fn_bodies = []
        self.symbol_table = SymbolTable()

        self._string_cache: dict[str, StaticVariable] = {}
        self._static_variables: list[StaticVariable] = []

        self._has_main: bool = False

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

        signature = self.symbol_table.lookup_function(
            fn_name, fn_specialization, fn_args
        )
        return FunctionCallExpression(signature, fn_args)

    def add_static_string(self, string: str) -> StaticVariable:
        if string in self._string_cache:
            return self._string_cache[string]

        encoded_str, encoded_length = encode_string(string)
        str_type = AnonymousType(CharArrayDefinition(encoded_str, encoded_length))
        static_storage = StaticVariable(
            f'string literal: "{string}"', str_type, True, encoded_str
        )
        self.add_static_variable(static_storage)

        self._string_cache[string] = static_storage
        return static_storage

    def add_static_variable(self, var: StaticVariable) -> None:
        self._static_variables.append(var)

    def add_function_body(self, function: Function) -> None:
        self._fn_bodies.append(function)

    def generate_ir(self) -> list[str]:
        lines: list[str] = []

        lines.append(f'target datalayout = "{target.get_datalayout()}"')
        lines.append(f'target triple = "{target.get_target_triple()}"')

        lines.append("")
        var_reg_gen = count(0)
        for variable in self._static_variables:
            lines.extend(variable.generate_ir(var_reg_gen))

        lines.append("")
        lines.extend(self.symbol_table.get_ir_for_initialization())

        lines.append("")
        for func in self._fn_bodies:
            lines.extend(func.generate_ir())

        return lines

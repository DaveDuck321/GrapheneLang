from collections import defaultdict
from functools import cached_property
from itertools import count

from .user_facing_errors import (
    FailedLookupError,
    OverloadResolutionError,
    RedefinitionError,
    assert_else_throw,
    throw,
)

from .interfaces import Type, TypedExpression, Variable
from .generatable import Scope, StackVariable, VariableAssignment
from .expressions import FunctionParameter, FunctionCallExpression
from .builtin_types import get_builtin_types, FunctionSignature
from .builtin_callables import get_builtin_callables


class Function:
    def __init__(
        self, name: str, parameters: list[Variable], return_type: Type, is_foreign: bool
    ) -> None:
        self._signature = FunctionSignature(
            name, [var.type for var in parameters], return_type, is_foreign
        )

        self.scope_id_iter = count()
        self.top_level_scope = Scope(self.get_next_scope_id())

        # Implicit stack variable allocation for parameters
        #   TODO: constant parameters (when/ if added to grammar)
        self._parameters: list[FunctionParameter] = []
        if not self.is_foreign():
            for var in parameters:
                fn_param = FunctionParameter(var.type)
                self._parameters.append(fn_param)
                self.top_level_scope.add_generatable(fn_param)

                fn_param_var = StackVariable(var.name, var.type, False, True)
                self.top_level_scope.add_variable(fn_param_var)

                fn_param_var_assignment = VariableAssignment(fn_param_var, fn_param)
                self.top_level_scope.add_generatable(fn_param_var_assignment)

    def __repr__(self) -> str:
        return repr(self._signature)

    @cached_property
    def mangled_name(self) -> str:
        return self._signature.mangled_name

    def get_signature(self) -> FunctionSignature:
        return self._signature

    def is_foreign(self) -> bool:
        return self._signature.is_foreign()

    def get_next_scope_id(self) -> int:
        return next(self.scope_id_iter)

    def generate_declaration(self) -> list[str]:
        ir = f"declare dso_local {self._signature.return_type.ir_type} @{self.mangled_name}("

        args_ir = [arg.ir_type for arg in self._signature.arguments]
        ir += str.join(", ", args_ir)

        # XXX nounwind indicates that the function never raises an exception.
        ir += ") nounwind"

        return [ir]

    def generate_definition(self) -> list[str]:
        lines: list[str] = []
        reg_gen = count(0)  # First register is %0

        for param in self._parameters:
            param.set_reg(next(reg_gen))

        args_ir = ", ".join(map(lambda param: param.ir_ref, self._parameters))

        lines.append(f"define dso_local i32 @{self.mangled_name}({args_ir}) {{")
        lines.append(f"begin:")  # Name the implicit basic block

        lines.extend(self.top_level_scope.generate_ir(reg_gen))

        lines.append("}")

        return lines

    def generate_ir(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#functions
        if self.is_foreign():
            assert not self.top_level_scope._lines
            return self.generate_declaration()

        return self.generate_definition()


class FunctionSymbolTable:
    def __init__(self) -> None:
        self.foreign_functions: list[Function] = []
        self.graphene_functions: list[Function] = []
        self._functions: dict[str, list[Function]] = defaultdict(list)

    def add_function(self, fn_to_add: Function) -> None:
        fn_to_add_signature = fn_to_add.get_signature()
        matched_functions = self._functions[fn_to_add_signature.name]

        def get_printable_sig(sig: FunctionSignature) -> str:
            return f"{sig.name}: ({', '.join(map(str, sig.arguments))})"

        if fn_to_add_signature.is_foreign() and len(matched_functions) > 0:
            RedefinitionError(
                "non-overloadable foreign function",
                get_printable_sig(fn_to_add_signature),
            )

        for target in matched_functions:
            assert_else_throw(
                target.is_foreign(),
                RedefinitionError(
                    "non-overloadable foreign function",
                    get_printable_sig(target.get_signature()),
                ),
            )
            assert_else_throw(
                target._signature.arguments != fn_to_add_signature.arguments,
                RedefinitionError("function", get_printable_sig(fn_to_add_signature)),
            )

        matched_functions.append(fn_to_add)
        if fn_to_add.is_foreign():
            self.foreign_functions.append(fn_to_add)
        else:
            self.graphene_functions.append(fn_to_add)

    def lookup_function(self, fn_name: str, fn_args: list[Type]):
        candidate_functions = self._functions[fn_name]

        readable_arg_names = ", ".join(map(repr, fn_args))
        assert_else_throw(
            len(candidate_functions) > 0,
            FailedLookupError(
                "function", f"function {fn_name}: ({readable_arg_names}) -> ..."
            ),
        )

        for function in candidate_functions:
            if function.get_signature().arguments == fn_args:
                return function

        throw(OverloadResolutionError(fn_name, readable_arg_names))


class Program:
    def __init__(self) -> None:
        super().__init__()
        self._builtin_callables = get_builtin_callables()

        self._function_table = FunctionSymbolTable()
        self._types: dict[str, Type] = {}
        self._strings: dict[str, str] = {}

        self._has_main: bool = False

        for type in get_builtin_types():
            self.add_type(type)

    def lookup_type(self, name: str) -> Type:
        assert_else_throw(
            name in self._types, FailedLookupError("type", f"typedef {name} : ...")
        )
        return self._types[name]

    def lookup_call_expression(
        self, fn_name: str, fn_args: list[TypedExpression]
    ) -> TypedExpression:
        if fn_name in self._builtin_callables:
            return self._builtin_callables[fn_name](fn_args)

        arg_types = [arg.type for arg in fn_args]
        function = self._function_table.lookup_function(fn_name, arg_types)
        return FunctionCallExpression(function.get_signature(), fn_args)

    def add_function(self, function: Function) -> None:
        self._function_table.add_function(function)

    def add_type(self, type: Type) -> None:
        assert_else_throw(
            type.name not in self._types, RedefinitionError("type", type.name)
        )
        self._types[type.name] = type

    @staticmethod
    def _get_string_identifier(index: int) -> str:
        assert index >= 0
        return f".str.{index}"

    def add_string(self, string: str) -> str:
        id = self._get_string_identifier(len(self._strings))
        self._strings[id] = string

        return id

    def generate_ir(self, target="x86_64-pc-linux-gnu") -> list[str]:
        lines: list[str] = []

        lines.append(f'target triple = "{target}"')

        for string_id, string in self._strings.items():
            # TODO encode string correctly
            lines.append(
                f'@{string_id} = private unnamed_addr constant [{len(string) + 1} x i8] c"{string}\\00"'
            )

        for fn in self._function_table.foreign_functions:
            lines.extend(fn.generate_ir())

        for fn in self._function_table.graphene_functions:
            lines.extend(fn.generate_ir())

        return lines

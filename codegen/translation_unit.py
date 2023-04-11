from collections import defaultdict
from functools import cached_property
from itertools import count
from typing import Callable, Iterable, Optional

from .builtin_callables import get_builtin_callables
from .builtin_types import CharArrayDefinition, FunctionSignature, get_builtin_types
from .expressions import FunctionCallExpression, FunctionParameter
from .generatable import Scope, StackVariable, StaticVariable, VariableAssignment
from .interfaces import Parameter, SpecializationItem, Type, TypedExpression
from .strings import encode_string
from .type_conversions import get_implicit_conversion_cost
from .user_facing_errors import (
    AmbiguousFunctionCall,
    FailedLookupError,
    OverloadResolutionError,
    RedefinitionError,
    SubstitutionFailure,
)


class Function:
    def __init__(
        self,
        name: str,
        parameters: list[Parameter],
        return_type: Type,
        is_foreign: bool,
        specialization: list[SpecializationItem],
    ) -> None:
        self._signature = FunctionSignature(
            name,
            [var.type for var in parameters],
            return_type,
            specialization,
            is_foreign,
        )

        self.scope_id_iter = count()
        self.top_level_scope = Scope(self.get_next_scope_id())

        # Implicit stack variable allocation for parameters
        #   TODO: constant parameters (when/ if added to grammar)
        self._parameters: list[FunctionParameter] = []
        if not self.is_foreign():
            for param in parameters:
                fn_param = FunctionParameter(param.type)
                self._parameters.append(fn_param)
                self.top_level_scope.add_generatable(fn_param)

                fn_param_var = StackVariable(param.name, param.type, False, True)
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

    def is_foreign(self) -> bool:
        return self._signature.is_foreign()

    def get_next_scope_id(self) -> int:
        return next(self.scope_id_iter)

    def generate_declaration(self) -> list[str]:
        args_ir = [arg.ir_type for arg in self._signature.arguments]

        # XXX nounwind indicates that the function never raises an exception.
        return [
            f"declare dso_local {self._signature.return_type.ir_type}"
            f" @{self.mangled_name}({str.join(', ', args_ir)}) nounwind"
        ]

    def generate_definition(self) -> list[str]:
        if not self.get_signature().return_type.is_void:
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

    def generate_ir(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#functions
        if self.is_foreign():
            assert self.top_level_scope.is_empty()
            return self.generate_declaration()

        return self.generate_definition()


class GenericFunctionParser:
    def __init__(
        self,
        name: str,
        parse_with_specialization_fn: Callable[
            [str, list[SpecializationItem]],
            Optional[Function],
        ],
        deduce_specialization_fn: Callable[
            [str, list[TypedExpression]], Optional[list[SpecializationItem]]
        ],
    ) -> None:
        self.fn_name = name
        self._parse_with_specialization_fn = parse_with_specialization_fn
        self._deduce_specialization_fn = deduce_specialization_fn

    def try_parse_with_specialization(
        self, specialization: list[SpecializationItem]
    ) -> Optional[Function]:
        return self._parse_with_specialization_fn(self.fn_name, specialization)

    def try_deduce_specialization(
        self, args: list[TypedExpression]
    ) -> Optional[list[SpecializationItem]]:
        return self._deduce_specialization_fn(self.fn_name, args)


class FunctionSymbolTable:
    def __init__(self) -> None:
        self.foreign_functions: list[Function] = []
        self.graphene_functions: list[Function] = []

        self._generic_parsers: dict[str, list[GenericFunctionParser]] = defaultdict(
            list
        )
        self._specialized_functions: dict[str, list[Function]] = defaultdict(list)
        self._functions: dict[str, list[Function]] = defaultdict(list)

    def add_generic_function(
        self,
        fn_name: str,
        parser: GenericFunctionParser,
    ) -> None:
        self._generic_parsers[fn_name].append(parser)

    def add_function(self, fn_to_add: Function) -> None:
        fn_to_add_signature = fn_to_add.get_signature()
        matched_functions = self._functions[fn_to_add_signature.name]

        if fn_to_add_signature.is_foreign() and len(matched_functions) > 0:
            raise RedefinitionError(
                "non-overloadable foreign function",
                fn_to_add_signature.user_facing_name,
            )

        for target in matched_functions:
            if target.is_foreign():
                raise RedefinitionError(
                    "non-overloadable foreign function",
                    target.get_signature().user_facing_name,
                )

            if target.get_signature().arguments == fn_to_add_signature.arguments:
                raise RedefinitionError(
                    "function", fn_to_add_signature.user_facing_name
                )

        matched_functions.append(fn_to_add)
        if fn_to_add.is_foreign():
            self.foreign_functions.append(fn_to_add)
        else:
            self.graphene_functions.append(fn_to_add)

    def generate_functions_with_specialization(
        self, fn_name: str, fn_specialization: list[SpecializationItem]
    ) -> list[Function]:
        # Have we already parsed this function?
        if fn_name in self._specialized_functions:
            matching_functions = []
            for candidate_function in self._specialized_functions[fn_name]:
                candidate_signature = candidate_function.get_signature()
                if candidate_signature.specialization == fn_specialization:
                    matching_functions.append(candidate_function)

            if matching_functions:
                return matching_functions

        # We have not parsed this function
        candidate_parsers = self._generic_parsers[fn_name]
        if len(candidate_parsers) == 0:
            raise NotImplementedError()

        matching_functions = []
        for parser in candidate_parsers:
            parsed_fn = parser.try_parse_with_specialization(fn_specialization)
            if parsed_fn is not None:
                matching_functions.append(parsed_fn)

        for matched_fn in matching_functions:
            # TODO: we don't actually need to generate all these functions, some might be unused
            self.graphene_functions.append(matched_fn)
            self._specialized_functions[fn_name].append(matched_fn)

        return matching_functions

    def lookup_explicitly_specialized_function(
        self,
        fn_name: str,
        fn_specialization: list[SpecializationItem],
        fn_args: list[TypedExpression],
    ) -> Function:
        matching_functions = self.generate_functions_with_specialization(
            fn_name, fn_specialization
        )
        return self.select_function_with_least_cost(
            fn_name, matching_functions, fn_specialization, fn_args
        )

    def lookup_function(self, fn_name: str, fn_args: list[TypedExpression]) -> Function:
        # The normal Graphene functions
        candidate_functions = [
            fn
            for fn in self._functions[fn_name]
            if len(fn.get_signature().specialization) == 0
        ]

        # Implicit generic instantiation
        candidate_specializations: list[list[SpecializationItem]] = []
        for candidate_generic in self._generic_parsers[fn_name]:
            fn_specialization = candidate_generic.try_deduce_specialization(fn_args)

            if (
                fn_specialization is not None
                and fn_specialization not in candidate_specializations
            ):
                candidate_specializations.append(fn_specialization)
                candidate_functions.extend(
                    self.generate_functions_with_specialization(
                        fn_name, fn_specialization
                    )
                )

        return self.select_function_with_least_cost(
            fn_name, candidate_functions, [], fn_args
        )

    def select_function_with_least_cost(
        self,
        fn_name: str,
        candidate_functions: Iterable[Function],
        fn_specialization: list[SpecializationItem],
        fn_args: list[TypedExpression],
    ) -> Function:
        functions_by_cost: list[tuple[int, Function]] = []

        for function in candidate_functions:
            total_cost = 0
            if len(fn_args) != len(function.get_signature().arguments):
                continue

            for src_type, dest_type in zip(fn_args, function.get_signature().arguments):
                cost = get_implicit_conversion_cost(src_type, dest_type)
                if cost is not None:
                    total_cost += cost
                else:
                    total_cost = None
                    break

            # Conversion might fail for some arguments. In that case, discard
            # this candidate.
            if total_cost is not None:
                functions_by_cost.append((total_cost, function))

        functions_by_cost.sort(key=lambda t: t[0])

        readable_arg_names = ", ".join(
            arg.get_user_facing_name(False) for arg in fn_args
        )
        readable_specialization = ", ".join(
            specialization.get_user_facing_name(False)
            if isinstance(specialization, Type)
            else str(specialization)
            for specialization in fn_specialization
        )

        if not functions_by_cost:
            raise OverloadResolutionError(
                fn_name,
                readable_specialization,
                readable_arg_names,
                [
                    fn.get_signature().user_facing_name
                    for fn in self._functions[fn_name]
                ],
            )

        if len(functions_by_cost) == 1:
            return functions_by_cost[0][1]

        first, second, *_ = functions_by_cost

        # If there are two or more equally good candidates, then this function
        # call is ambiguous.
        if first[0] == second[0]:
            raise AmbiguousFunctionCall(
                fn_name,
                readable_arg_names,
                first[1].get_signature().user_facing_name,
                second[1].get_signature().user_facing_name,
            )

        return first[1]


class Program:
    def __init__(self) -> None:
        super().__init__()
        self._builtin_callables = get_builtin_callables()

        self._function_table = FunctionSymbolTable()
        self._initialized_types: dict[str, list[Type]] = defaultdict(list)
        self._type_initializers: dict[
            str, Callable[[str, list[SpecializationItem]], Type]
        ] = {}

        self._string_cache: dict[str, StaticVariable] = {}
        self._static_variables: list[StaticVariable] = []

        self._has_main: bool = False

        for builtin_type in get_builtin_types():
            name = builtin_type.get_user_facing_name(False)
            self._initialized_types[name].append(builtin_type)

    def lookup_type(
        self, name_prefix: str, generic_args: list[SpecializationItem]
    ) -> Type:
        specialization_str = ", ".join(
            arg.get_user_facing_name(False) if isinstance(arg, Type) else str(arg)
            for arg in generic_args
        )
        specialization_postfix = f"<{specialization_str}>" if specialization_str else ""

        if (
            name_prefix not in self._initialized_types
            and name_prefix not in self._type_initializers
        ):
            # No typedefs or specializations: type always fails SFINAE
            raise FailedLookupError(
                "type", f"typedef {name_prefix}{specialization_postfix} : ..."
            )

        if name_prefix in self._initialized_types:
            # Reuse parsed type if it is already initialized
            matching_candidates = []
            for candidate_type in self._initialized_types[name_prefix]:
                if generic_args == candidate_type.get_specialization():
                    matching_candidates.append(candidate_type)

            if len(matching_candidates) == 1:
                return matching_candidates[0]

            assert len(matching_candidates) == 0

        # Type is not already initialized
        if name_prefix not in self._type_initializers:
            # Substitution failure is an error here :-D
            raise SubstitutionFailure(f"{name_prefix}{specialization_postfix}")

        this_type = self._type_initializers[name_prefix](name_prefix, generic_args)
        self._initialized_types[name_prefix].append(this_type)
        return this_type

    def lookup_call_expression(
        self,
        fn_name: str,
        fn_specialization: list[SpecializationItem],
        fn_args: list[TypedExpression],
    ) -> TypedExpression:
        if fn_name in self._builtin_callables:
            return self._builtin_callables[fn_name](fn_specialization, fn_args)

        if len(fn_specialization) != 0:
            function = self._function_table.lookup_explicitly_specialized_function(
                fn_name, fn_specialization, fn_args
            )
        else:
            function = self._function_table.lookup_function(fn_name, fn_args)

        return FunctionCallExpression(function.get_signature(), fn_args)

    def add_generic_function(self, fn_name: str, parser: GenericFunctionParser):
        self._function_table.add_generic_function(fn_name, parser)

    def add_function(self, function: Function) -> None:
        self._function_table.add_function(function)

    def add_type(
        self,
        type_prefix: str,
        parse_fn: Callable[[str, list[SpecializationItem]], Type],
    ) -> None:
        if type_prefix in self._type_initializers:
            raise RedefinitionError("type", type_prefix)

        if Type.mangle_generic_type(type_prefix, []) in self._type_initializers:
            raise RedefinitionError("builtin type", type_prefix)

        self._type_initializers[type_prefix] = parse_fn

    def add_specialized_type(
        self,
        type_prefix: str,
        parse_fn: Callable[[str, list[SpecializationItem]], Type],
        specialization: list[SpecializationItem],
    ) -> None:
        parsed_type = parse_fn(type_prefix, specialization)
        existing_type_specializations = self._initialized_types[type_prefix]

        for existing_type in existing_type_specializations:
            if existing_type.get_specialization() == specialization:
                raise RedefinitionError(
                    "specialized type", parsed_type.get_user_facing_name(False)
                )

        self._initialized_types[type_prefix].append(parsed_type)

    def add_static_string(self, string: str) -> StaticVariable:
        if string in self._string_cache:
            return self._string_cache[string]

        encoded_str, encoded_length = encode_string(string)
        str_type = Type(CharArrayDefinition(encoded_str, encoded_length))
        static_storage = StaticVariable(str_type, True, encoded_str)
        self.add_static_variable(static_storage)

        self._string_cache[string] = static_storage
        return static_storage

    def add_static_variable(self, var: StaticVariable) -> None:
        self._static_variables.append(var)

    def generate_ir(self, target="x86_64-pc-linux-gnu") -> list[str]:
        lines: list[str] = []

        lines.append(f'target triple = "{target}"')

        lines.append("")
        var_reg_gen = count(0)
        for variable in self._static_variables:
            lines.extend(variable.generate_ir(var_reg_gen))

        lines.append("")
        for type_family in self._initialized_types.values():
            for defined_type in type_family:
                lines.extend(defined_type.get_ir_initial_type_def())

        lines.append("")
        for func in self._function_table.foreign_functions:
            lines.extend(func.generate_ir())

        lines.append("")
        for func in self._function_table.graphene_functions:
            lines.extend(func.generate_ir())
            lines.append("")

        return lines

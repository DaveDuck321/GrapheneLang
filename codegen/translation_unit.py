from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, reduce
from itertools import count
from typing import Callable, Iterator

from .builtin_callables import get_builtin_callables
from .builtin_types import FunctionSignature, get_builtin_types
from .expressions import FunctionCallExpression, FunctionParameter
from .generatable import Scope, StackVariable, VariableAssignment
from .interfaces import Parameter, Type, TypedExpression
from .type_conversions import get_implicit_conversion_cost
from .user_facing_errors import (
    AmbiguousFunctionCall,
    FailedLookupError,
    InvalidEscapeSequence,
    OverloadResolutionError,
    RedefinitionError,
    assert_else_throw,
    throw,
)


class Function:
    def __init__(
        self,
        name: str,
        parameters: list[Parameter],
        return_type: Type,
        is_foreign: bool,
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
        ir = (
            f"declare dso_local {self._signature.return_type.ir_type}"
            f" @{self.mangled_name}("
        )

        args_ir = [arg.ir_type for arg in self._signature.arguments]
        ir += str.join(", ", args_ir)

        # XXX nounwind indicates that the function never raises an exception.
        ir += ") nounwind"

        return [ir]

    def generate_definition(self) -> list[str]:
        reg_gen = count(0)  # First register is %0

        for param in self._parameters:
            param.set_reg(next(reg_gen))

        args_ir = ", ".join(
            map(lambda param: param.ir_ref_with_type_annotation, self._parameters)
        )

        def indent_ir(ir: list[str]):
            return map(lambda line: line if line.endswith(":") else f"  {line}", ir)

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
                not target.is_foreign(),
                RedefinitionError(
                    "non-overloadable foreign function",
                    get_printable_sig(target.get_signature()),
                ),
            )
            assert_else_throw(
                target.get_signature().arguments != fn_to_add_signature.arguments,
                RedefinitionError("function", get_printable_sig(fn_to_add_signature)),
            )

        matched_functions.append(fn_to_add)
        if fn_to_add.is_foreign():
            self.foreign_functions.append(fn_to_add)
        else:
            self.graphene_functions.append(fn_to_add)

    def lookup_function(self, fn_name: str, fn_args: list[Type]):
        # FIXME check argument number.
        candidate_functions = self._functions[fn_name]

        readable_arg_names = ", ".join(
            arg.get_user_facing_name(False) for arg in fn_args
        )
        assert_else_throw(
            len(candidate_functions) > 0,
            FailedLookupError(
                "function", f"function {fn_name}: ({readable_arg_names}) -> ..."
            ),
        )

        functions_by_cost: list[tuple[tuple[int, int], Function]] = []
        tuple_add = lambda a, b: tuple(sum(pair) for pair in zip(a, b))

        for function in candidate_functions:
            per_arg = list(
                map(
                    get_implicit_conversion_cost,
                    fn_args,
                    function.get_signature().arguments,
                )
            )

            # Conversion failed for at least one argument, discard the candidate
            # function.
            if any(cost is None for cost in per_arg):
                continue

            costs = reduce(tuple_add, per_arg, (0, 0))
            assert costs is not None

            functions_by_cost.append((costs, function))

        # List is sorted by the first element, then by the second.
        functions_by_cost.sort(key=lambda t: t[0])

        if len(functions_by_cost) == 1:
            return functions_by_cost[0][1]

        if len(functions_by_cost) > 1:
            first, second = functions_by_cost[:2]

            # If there's two or more equally good candidates, then this function
            # call is ambiguous.
            assert_else_throw(
                first[0] < second[0],
                AmbiguousFunctionCall(
                    fn_name,
                    readable_arg_names,
                    first[1].get_signature().user_facing_name,
                    second[1].get_signature().user_facing_name,
                ),
            )

            return first[1]

        throw(OverloadResolutionError(fn_name, readable_arg_names))


@dataclass
class StringInfo:
    identifier: str
    encoded_string: str
    encoded_length: int

    def __iter__(self) -> Iterator:
        # Support for unpack the dataclass.
        return iter((self.identifier, self.encoded_string, self.encoded_length))


class Program:
    # Based on https://en.cppreference.com/w/cpp/language/escape.
    # We omit \' because we don't have single-quoted strings, and \? because
    # we don't have trigraphs.
    ESCAPE_SEQUENCES_TABLE = {
        '"': chr(0x22),
        "\\": chr(0x5C),
        "a": chr(0x07),
        "b": chr(0x08),
        "f": chr(0x0C),
        "n": chr(0x0A),
        "r": chr(0x0D),
        "t": chr(0x09),
        "v": chr(0x0B),
    }

    def __init__(self) -> None:
        super().__init__()
        self._builtin_callables = get_builtin_callables()

        self._function_table = FunctionSymbolTable()
        self._types: dict[str, Type] = {}
        self._type_initializers: dict[str, Callable[[str, list[Type]], Type]] = {}
        self._strings: dict[str, StringInfo] = {}

        self._has_main: bool = False

        for builtin_type in get_builtin_types():
            self._types[builtin_type.mangled_name] = builtin_type

    def lookup_type(self, name_prefix: str, generic_args: list[Type]) -> Type:
        this_mangle = Type.mangle_generic_type(name_prefix, generic_args)
        if this_mangle in self._types:
            return self._types[this_mangle]

        assert_else_throw(
            name_prefix in self._type_initializers,
            FailedLookupError("type", f"typedef {name_prefix}[...] : ..."),
        )

        this_type = self._type_initializers[name_prefix](name_prefix, generic_args)
        self._types[this_mangle] = this_type
        return this_type

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

    def add_type(
        self, type_prefix: str, parse_fn: Callable[[str, list[Type]], Type]
    ) -> None:
        assert_else_throw(
            type_prefix not in self._type_initializers,
            RedefinitionError("type", type_prefix),
        )
        assert_else_throw(
            Type.mangle_generic_type(type_prefix, []) not in self._type_initializers,
            RedefinitionError("builtin type", type_prefix),
        )
        self._type_initializers[type_prefix] = parse_fn

    @staticmethod
    def _get_string_identifier(index: int) -> str:
        assert index >= 0
        return f"@.str.{index}"

    def add_string(self, string: str) -> str:
        # Deduplicate string constants.
        if string in self._strings:
            return self._strings[string].identifier

        # Encoding the string now means that we can provide line information in
        # case of an error.
        encoded_string, encoded_length = self.encode_string(string)
        identifier = self._get_string_identifier(len(self._strings))

        self._strings[string] = StringInfo(
            identifier,
            encoded_string,
            encoded_length,
        )

        return identifier

    @classmethod
    def encode_string(cls, string: str) -> tuple[str, int]:
        # LLVM is a bit vague on what is acceptable, but we definitely need to
        # escape non-printable characters and double quotes with "\xx", where
        # xx is the hexadecimal representation of each byte. We also parse
        # escape sequences here.
        # XXX we're using utf-8 for everything.

        first: int = 0
        byte_len: int = 0
        buffer: list[str] = []

        def encode_char(char: str) -> None:
            nonlocal byte_len

            utf8_bytes = char.encode("utf-8")

            for byte in utf8_bytes:
                # We can't store zeros in a null-terminated string.
                assert byte != 0

                buffer.append(f"\\{byte:0>2x}")

            byte_len += len(utf8_bytes)

        def consume_substr(last: int) -> None:
            nonlocal byte_len

            # Append the substr as-is instead of the utf-8 representation, as
            # python will encode it anyway when we write to the output stream.
            substr = string[first:last]
            buffer.append(substr)
            # FIXME there must be a better way.
            byte_len += len(substr.encode("utf-8"))

        chars = iter(enumerate(string))
        for idx, char in chars:
            if char == "\\":
                # Consume up to the previous character.
                consume_substr(idx)

                # Should never raise StopIteration, as the grammar guarantees
                # that we don't end a screen with a \.
                _, escaped_char = next(chars)

                assert_else_throw(
                    escaped_char in cls.ESCAPE_SEQUENCES_TABLE,
                    InvalidEscapeSequence(escaped_char),
                )

                # Easier if we always encode the representation of the escape
                # sequence.
                encode_char(cls.ESCAPE_SEQUENCES_TABLE[escaped_char])

                # Start from the character after the next one.
                first = idx + 2
            elif not char.isprintable():
                # Consume up to the previous character.
                consume_substr(idx)

                # Escape current character.
                encode_char(char)

                # Start from the next character.
                first = idx + 1

        # Consume any remaining characters.
        consume_substr(len(string))

        # Append the null terminator.
        buffer.append("\\00")
        byte_len += 1

        # Appending to a str in a loop is O(n^2).
        return str.join("", buffer), byte_len

    def generate_ir(self, target="x86_64-pc-linux-gnu") -> list[str]:
        lines: list[str] = []

        lines.append(f'target triple = "{target}"')

        lines.append("")
        for identifier, encoded_string, encoded_length in self._strings.values():
            lines.append(
                f"{identifier} = private unnamed_addr constant"
                f' [{encoded_length} x i8] c"{encoded_string}"'
            )

        lines.append("")
        for type in self._types.values():
            lines.extend(type.get_ir_initial_type_def())

        lines.append("")
        for func in self._function_table.foreign_functions:
            lines.extend(func.generate_ir())

        lines.append("")
        for func in self._function_table.graphene_functions:
            lines.extend(func.generate_ir())
            lines.append("")

        return lines

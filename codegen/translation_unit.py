from collections import defaultdict
from functools import cached_property
from itertools import count

from .user_facing_errors import (
    FailedLookupError,
    OverloadResolutionError,
    RedefinitionError,
    InvalidEscapeSequence,
    assert_else_throw,
    throw,
)

from .builtin_callables import get_builtin_callables
from .builtin_types import get_builtin_types, FunctionSignature
from .expressions import FunctionParameter, FunctionCallExpression
from .generatable import Scope, StackVariable, VariableAssignment
from .interfaces import Type, TypedExpression, Variable


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

        def indent_ir(ir: list[str]):
            return [f"  {line}" for line in ir]

        return [
            f"define dso_local {self._signature.return_type.ir_type} @{self.mangled_name}({args_ir}) {{",
            "begin:",  # Name the implicit basic block
            *indent_ir(self.top_level_scope.generate_ir(reg_gen)),
            "}",
        ]

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
            if all(
                arg.is_implicitly_convertible_to(other_arg)
                for arg, other_arg in zip(fn_args, function.get_signature().arguments)
            ):
                return function

        throw(OverloadResolutionError(fn_name, readable_arg_names))


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
        self._strings: dict[str, tuple[str, int]] = {}

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

        # Encoding the string now means that we can provide line information in
        # case of an error.
        self._strings[id] = self.encode_string(string)

        return id

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
        for string_id, (string, length) in self._strings.items():
            lines.append(
                f'@{string_id} = private unnamed_addr constant [{length} x i8] c"{string}"'
            )

        lines.append("")
        for type in self._types.values():
            lines.extend(type.get_definition_ir())

        lines.append("")
        for fn in self._function_table.foreign_functions:
            lines.extend(fn.generate_ir())

        lines.append("")
        for fn in self._function_table.graphene_functions:
            lines.extend(fn.generate_ir())
            lines.append("")

        return lines

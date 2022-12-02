from typing import Any
from dataclasses import dataclass
from functools import cached_property

from .interfaces import Type


class IntType(Type):
    align = 4
    ir_type = "i32"

    def __init__(self) -> None:
        super().__init__("int", "__builtin_int")

    def compatible_with(self, value: Any) -> bool:
        # TODO check if value fits inside an i32
        return isinstance(value, int)

    def cast_constant(self, value: int) -> int:
        assert self.compatible_with(value)
        return int(value)


class BoolType(Type):
    align = 1
    ir_type = "i1"

    def __init__(self) -> None:
        super().__init__("bool", "__builtin_bool")

    def compatible_with(self, value: Any) -> bool:
        return isinstance(value, bool)

    def cast_constant(self, value: bool) -> int:
        assert self.compatible_with(value)
        return int(value)


class StringType(Type):
    align = 1
    ir_type = "ptr"

    def __init__(self) -> None:
        super().__init__("string", "__builtin_str")

    def compatible_with(self, value: Any) -> bool:
        return isinstance(value, str)

    def cast_constant(self, value: str) -> str:
        assert self.compatible_with(value)
        return str(value)


@dataclass
class FunctionSignature:
    name: str
    arguments: list[Type]
    return_type: Type
    foreign: bool = False

    def is_main(self) -> bool:
        return self.name == "main"

    def is_foreign(self) -> bool:
        return self.foreign

    @cached_property
    def mangled_name(self) -> str:
        # main() is immune to name mangling (irrespective of arguments)
        if self.is_main() or self.is_foreign():
            return self.name

        arguments_mangle = [arg.mangled_name for arg in self.arguments]

        # FIXME separator
        arguments_mangle = "".join(arguments_mangle)

        # Name mangle operators into digits
        legal_name_mangle = []
        for char in self.name:
            if char.isalnum():
                legal_name_mangle.append(char)
            else:
                legal_name_mangle.append(f"__O{ord(char)}")

        return f"__{''.join(legal_name_mangle)}__ARGS__{arguments_mangle}"

    def __repr__(self) -> str:
        readable_arg_names = ", ".join(map(repr, self.arguments))
        if self.is_foreign():
            return f"foreign {self.name}: ({readable_arg_names}) -> {self.return_type.name}"
        else:
            return f"function {self.name}: ({readable_arg_names}) -> {self.return_type.name}"

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.return_type.ir_type} @{self.mangled_name}"


def get_builtin_types() -> list[Type]:
    return [IntType(), StringType(), BoolType()]

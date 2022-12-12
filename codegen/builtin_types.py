from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .interfaces import Parameter, Type, TypeDefinition
from .user_facing_errors import FailedLookupError, throw


class PrimitiveDefinition(TypeDefinition):
    align = 0
    ir = ""

    def get_alignment(self) -> int:
        assert self.align != 0
        return self.align

    def get_anonymous_ir_ref(self) -> str:
        assert self.ir != ""
        return self.ir

    def get_named_ir_ref(self, name) -> str:
        return f"%primitive.{name}"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        return isinstance(other, type(self))


class IntType(Type):
    class Definition(PrimitiveDefinition):
        align = 4
        ir = "i32"

        def __init__(self) -> None:
            super().__init__()

        def compatible_with(self, value: Any) -> bool:
            # TODO check if value fits inside an i32
            return isinstance(value, int)

        def cast_constant(self, value: int) -> int:
            assert self.compatible_with(value)
            return int(value)

    def __init__(self) -> None:
        super().__init__(self.Definition(), "int")


class BoolType(Type):
    class Definition(PrimitiveDefinition):
        align = 1
        ir = "i1"

        def __init__(self) -> None:
            super().__init__()

        def compatible_with(self, value: Any) -> bool:
            return isinstance(value, bool)

        def cast_constant(self, value: bool) -> int:
            assert self.compatible_with(value)
            return int(value)

    def __init__(self) -> None:
        super().__init__(self.Definition(), "bool")


class StringType(Type):
    class Definition(PrimitiveDefinition):
        align = 1
        ir = "ptr"

        def __init__(self) -> None:
            super().__init__()

        def compatible_with(self, value: Any) -> bool:
            return isinstance(value, str)

        def cast_constant(self, value: str) -> str:
            assert self.compatible_with(value)
            return str(value)

    def __init__(self) -> None:
        super().__init__(self.Definition(), "string")


class ReferenceType(Type):
    is_reference = True

    class Definition(PrimitiveDefinition):
        align = 8  # FIXME maybe we shouldn't hardcode pointer alignment.
        ir = "ptr"

        def __init__(self, value_type: Type) -> None:
            super().__init__()

            self.value_type = value_type

        def compatible_with(self, value: Any) -> bool:
            raise NotImplementedError("ReferenceType.compatible_with")

        def cast_constant(self, value: int) -> bool:
            raise NotImplementedError("ReferenceType.cast_constant")

        def get_named_ir_ref(self, _: str) -> str:
            # Opaque pointer type.
            return self.get_anonymous_ir_ref()

        def get_non_reference_type(self) -> Type:
            return self.value_type

    def __init__(self, value_type: Type) -> None:
        super().__init__(self.Definition(value_type), f"{value_type}&")


class StructDefinition(TypeDefinition):
    def __init__(self, members: list[Parameter]) -> None:
        super().__init__()

        # TODO: assert member names are unique
        self._members = members

    def get_member(self, name: str) -> tuple[int, Type]:
        for index, member in enumerate(self._members):
            if member.name == name:
                return index, member.type
        throw(FailedLookupError("struct member", f"{{{name}: ...}}"))

    def compatible_with(self, value: Any) -> bool:
        raise NotImplementedError()

    def cast_constant(self, value: int) -> bool:
        raise NotImplementedError()

    def get_anonymous_ir_ref(self) -> str:
        member_ir = [member.type.ir_type for member in self._members]
        return f"{{{', '.join(member_ir)}}}"

    def get_named_ir_ref(self, name) -> str:
        return f"%struct.{name}"

    def get_alignment(self) -> int:
        # TODO: can we be less conservative here
        return max(member.type.align for member in self._members)

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        if not isinstance(other, StructDefinition):
            return False
        return self._members == other._members


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
            return f"foreign {self.name}: ({readable_arg_names}) -> {repr(self.return_type)}"
        else:
            return f"function {self.name}: ({readable_arg_names}) -> {repr(self.return_type)}"

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.return_type.ir_type} @{self.mangled_name}"


def get_builtin_types() -> list[Type]:
    return [IntType(), StringType(), BoolType()]

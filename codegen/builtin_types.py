from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .interfaces import Parameter, Type, TypeDefinition
from .user_facing_errors import (
    assert_else_throw,
    throw,
    FailedLookupError,
    InvalidIntSize,
)


class PrimitiveDefinition(TypeDefinition):
    align = 0
    ir = ""
    inbuilt_name = ""

    def get_alignment(self) -> int:
        assert self.align != 0
        return self.align

    def mangled_name_for_ir(self) -> str:
        # Assert not reached:
        #   it should be impossible to create an anonymous primitive type
        #   ref/ structs overwrite this
        assert False

    def get_anonymous_ir_type_def(self) -> str:
        assert self.ir != ""
        return self.ir

    def get_named_ir_type_ref(self, name) -> str:
        return f"%alias.{name}"

    def __repr__(self) -> str:
        return f"Typedef({self.inbuilt_name})"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        if isinstance(other, PrimitiveDefinition):
            # TODO: I'm not sure this is the correct approach
            return (
                self.align == other.align
                and self.inbuilt_name == other.inbuilt_name
                and self.ir == other.ir
            )
        return False


def get_integral_definition(
    name: str, size_in_bits: int, is_signed: bool
) -> PrimitiveDefinition:
    ir_type = f"i{size_in_bits}"
    alignment = size_in_bits // 8

    class Definition(PrimitiveDefinition):
        align = alignment
        ir = ir_type
        inbuilt_name = name

        def to_ir_constant(self, value: str) -> str:
            if is_signed:
                range_lower = -(2 ** (size_in_bits - 1))
                range_upper = 2 ** (size_in_bits - 1)
            else:
                range_lower = 0
                range_upper = 2**size_in_bits

            assert_else_throw(
                range_lower <= int(value) < range_upper,
                InvalidIntSize(name, int(value), range_lower, range_upper),
            )
            return value

    return Definition()


class GenericIntType(Type):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        is_power_of_2 = ((size_in_bits - 1) & size_in_bits) == 0
        is_divisible_into_bytes = (size_in_bits % 8) == 0
        assert is_power_of_2 and is_divisible_into_bytes

        definition = get_integral_definition(name, size_in_bits, is_signed)
        super().__init__(definition, definition.inbuilt_name)


class IntType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("int", 32, True)


class BoolType(Type):
    class Definition(PrimitiveDefinition):
        align = 1
        ir = "i1"
        inbuilt_name = "bool"

        def to_ir_constant(self, value: str) -> str:
            # We happen to be using the same boolean constants as LLVM IR.
            assert value == "true" or value == "false"

            return value

    def __init__(self) -> None:
        definition = self.Definition()
        super().__init__(definition, definition.inbuilt_name)


class StringType(Type):
    class Definition(PrimitiveDefinition):
        align = 1
        ir = "ptr"
        inbuilt_name = "string"

        def to_ir_constant(self, identifier: str) -> str:
            # String constants are handled at the translation unit level.
            assert identifier.startswith("@.str.")

            return identifier

    def __init__(self) -> None:
        definition = self.Definition()
        super().__init__(definition, definition.inbuilt_name)


class ReferenceType(Type):
    is_reference = True

    class Definition(PrimitiveDefinition):
        align = 8  # FIXME maybe we shouldn't hardcode pointer alignment.
        ir = "ptr"

        def __init__(self, value_type: Type) -> None:
            super().__init__()

            self.value_type = value_type

        @cached_property
        def mangled_name_for_ir(self) -> str:
            return f"__RT{self.value_type.mangled_name_for_ir}__TR"

        def get_named_ir_type_ref(self, _: str) -> str:
            # Opaque pointer type.
            return self.get_anonymous_ir_type_def()

        def to_ir_constant(self, value: str) -> str:
            # We shouldn't be able to initialize a reference with a constant.
            assert False

    def get_non_reference_type(self) -> Type:
        assert isinstance(self.definition, self.Definition)
        return self.definition.value_type

    def __init__(self, value_type: Type) -> None:
        super().__init__(self.Definition(value_type))


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

    def mangled_name_for_ir(self) -> str:
        subtypes = [member.type.mangled_name_for_ir for member in self._members]
        return f"__ST{''.join(subtypes)}__TS"

    def to_ir_constant(self, value: str) -> str:
        # TODO support structure constants.
        raise NotImplementedError()

    def get_anonymous_ir_type_def(self) -> str:
        member_ir = [member.type.ir_type_annotation for member in self._members]
        return f"{{{', '.join(member_ir)}}}"

    def get_named_ir_type_ref(self, name) -> str:
        return f"%struct.{name}"

    def get_alignment(self) -> int:
        # TODO: can we be less conservative here
        return max(member.type.get_alignment() for member in self._members)

    def __repr__(self) -> str:
        return f"StructDefinition({', '.join(map(repr, self._members))})"

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

        arguments_mangle = [arg.mangled_name_for_ir for arg in self.arguments]

        # FIXME separator
        arguments_mangle = "".join(arguments_mangle)

        # Name mangle operators into digits
        legal_name_mangle = []
        for char in self.name:
            if char.isalnum():
                legal_name_mangle.append(char)
            else:
                legal_name_mangle.append(f"__O{ord(char)}")

        return f"{''.join(legal_name_mangle)}{arguments_mangle}"

    def __repr__(self) -> str:
        readable_arg_names = ", ".join(map(repr, self.arguments))
        if self.is_foreign():
            return f"foreign {self.name}: ({readable_arg_names}) -> {repr(self.return_type)}"
        else:
            return f"function {self.name}: ({readable_arg_names}) -> {repr(self.return_type)}"

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.return_type.ir_type_annotation} @{self.mangled_name}"


def get_builtin_types() -> list[Type]:
    # Generate sized int types
    sized_int_types = []
    for size in (8, 16, 32, 64, 128):
        sized_int_types.append(GenericIntType(f"i{size}", size, True))
        sized_int_types.append(GenericIntType(f"u{size}", size, False))

    return sized_int_types + [IntType(), StringType(), BoolType()]

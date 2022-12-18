from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Optional

from .interfaces import Parameter, Type, TypeDefinition
from .user_facing_errors import (
    FailedLookupError,
    InvalidIntSize,
    assert_else_throw,
    throw,
)


class PrimitiveDefinition(TypeDefinition):
    def __init__(self, align: int, ir: str, name: str) -> None:
        super().__init__()

        assert align > 0
        assert ir
        assert name

        self._align = align
        self._ir = ir
        self._name = name

    @cached_property
    def alignment(self) -> int:
        return self._align

    @cached_property
    def user_facing_name(self) -> str:
        return self._name

    def get_ir_type(self, alias: Optional[str]) -> str:
        return f"%alias.{alias}" if alias else self.ir_definition

    @cached_property
    def ir_definition(self) -> str:
        return self._ir

    @cached_property
    def mangled_name(self) -> str:
        # Assert not reached:
        #   it should be impossible to create an anonymous primitive type
        #   ref/ structs overwrite this
        assert False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        if isinstance(other, PrimitiveDefinition):
            # TODO: I'm not sure this is the correct approach
            return self._align == other._align and self._ir == other._ir

        return False


class IntegerDefinition(PrimitiveDefinition):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        super().__init__(size_in_bits // 8, f"i{size_in_bits}", name)

        self.is_signed = is_signed
        # TODO: maybe PrimitiveDefinition should have a size property.
        self.bits = size_in_bits

    def to_ir_constant(self, value: str) -> str:
        if self.is_signed:
            range_lower = -(2 ** (self.bits - 1))
            range_upper = 2 ** (self.bits - 1)
        else:
            range_lower = 0
            range_upper = 2**self.bits

        assert_else_throw(
            range_lower <= int(value) < range_upper,
            InvalidIntSize(self._name, int(value), range_lower, range_upper),
        )

        return value

    def __eq__(self, other: Any) -> bool:
        if not super().__eq__(other):
            return False

        if isinstance(other, IntegerDefinition):
            return self.is_signed == other.is_signed and self.bits == other.bits

        return False


class GenericIntType(Type):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        is_power_of_2 = ((size_in_bits - 1) & size_in_bits) == 0
        is_divisible_into_bytes = (size_in_bits % 8) == 0
        assert is_power_of_2 and is_divisible_into_bytes

        definition = IntegerDefinition(name, size_in_bits, is_signed)
        super().__init__(definition, definition._name)


class IntType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("int", 32, True)


class BoolType(Type):
    class Definition(PrimitiveDefinition):
        def __init__(self) -> None:
            super().__init__(1, "i1", "bool")

        def to_ir_constant(self, value: str) -> str:
            # We happen to be using the same boolean constants as LLVM IR.
            assert value in ("true", "false")

            return value

    def __init__(self) -> None:
        definition = self.Definition()
        super().__init__(definition, definition._name)


class StringType(Type):
    class Definition(PrimitiveDefinition):
        def __init__(self) -> None:
            super().__init__(8, "ptr", "string")

        def to_ir_constant(self, value: str) -> str:
            # String constants are handled at the translation unit level. We
            # should already have an identifier.
            assert value.startswith("@.str.")

            return value

    def __init__(self) -> None:
        definition = self.Definition()
        super().__init__(definition, definition._name)


class AddressableTypeDefinition(TypeDefinition):
    def __init__(self, value_type: Type) -> None:
        super().__init__()

        self.value_type = value_type

    def to_ir_constant(self, _: str) -> str:
        # We shouldn't be able to initialize a reference with a constant.
        assert False

    @cached_property
    def alignment(self) -> int:
        # FIXME replace magic number. References are pointer-aligned.
        return 8

    def get_ir_type(self, _: Optional[str]) -> str:
        # Opaque pointer type.
        return self.ir_definition

    @cached_property
    def ir_definition(self) -> str:
        return "ptr"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)

        if isinstance(other, type(self)):
            return self.value_type == other.value_type

        return False


class AddressableValueType(Type):
    is_reference = True

    class Definition(AddressableTypeDefinition):
        @cached_property
        def mangled_name(self) -> str:
            assert False

        @cached_property
        def user_facing_name(self) -> str:
            return f"{self.value_type.get_user_facing_name(False)}"

        def __repr__(self) -> str:
            return f"AddressableValue.Definition({repr(self.value_type)})"

    def get_non_reference_type(self) -> Type:
        assert isinstance(self.definition, self.Definition)
        return self.definition.value_type

    def __init__(self, value_type: Type) -> None:
        super().__init__(self.Definition(value_type))


class ReferenceType(Type):
    is_reference = True

    class Definition(AddressableTypeDefinition):
        @cached_property
        def mangled_name(self) -> str:
            return f"__RT{self.value_type.mangled_name}__TR"

        @cached_property
        def user_facing_name(self) -> str:
            return f"{self.value_type.get_user_facing_name(False)}&"

        def __repr__(self) -> str:
            return f"ReferenceType.Definition({repr(self.value_type)})"

    def get_non_reference_type(self) -> Type:
        assert isinstance(self.definition, self.Definition)
        return self.definition.value_type

    def __init__(self, value_type: Type, is_borrowed: bool) -> None:
        super().__init__(self.Definition(value_type), is_borrowed=is_borrowed)


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

    def to_ir_constant(self, value: str) -> str:
        # TODO support structure constants.
        raise NotImplementedError()

    @cached_property
    def user_facing_name(self) -> str:
        subtypes = map(lambda m: m.type.get_user_facing_name(False), self._members)
        return "{" + str.join(", ", subtypes) + "}"

    def get_ir_type(self, alias: Optional[str]) -> str:
        # If this type has an alias, then we've generated a definition for it at
        # the top of the IR source.
        if alias:
            return f"%struct.{alias}"

        # If it doesn't, then we need to define the type here.
        return self.ir_definition

    @cached_property
    def ir_definition(self) -> str:
        member_ir = map(lambda m: m.type.ir_type, self._members)
        return "{" + str.join(", ", member_ir) + "}"

    @cached_property
    def mangled_name(self) -> str:
        subtypes = map(lambda m: m.type.mangled_name, self._members)
        return f"__ST{''.join(subtypes)}__TS"

    @cached_property
    def alignment(self) -> int:
        # TODO: can we be less conservative here
        return max(member.type.alignment for member in self._members)

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

        return f"{''.join(legal_name_mangle)}{arguments_mangle}"

    def _repr_impl(self, key: Callable[[Type], str]) -> str:
        prefix = "foreign" if self.is_foreign() else "function"

        readable_arg_names = str.join(", ", map(key, self.arguments))

        return (
            f"{prefix} {self.name}: ({readable_arg_names}) -> "
            f"{key(self.return_type)}"
        )

    def __repr__(self) -> str:
        return self._repr_impl(repr)

    @cached_property
    def user_facing_name(self) -> str:
        return self._repr_impl(lambda t: t.get_user_facing_name(False))

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.return_type.ir_type} @{self.mangled_name}"


def get_builtin_types() -> list[Type]:
    # Generate sized int types
    sized_int_types = []
    for size in (8, 16, 32, 64, 128):
        sized_int_types.append(GenericIntType(f"i{size}", size, True))
        sized_int_types.append(GenericIntType(f"u{size}", size, False))

    return sized_int_types + [IntType(), StringType(), BoolType()]

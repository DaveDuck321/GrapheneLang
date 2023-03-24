import uuid
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import mul
from typing import Any, Callable, Optional

from .interfaces import (
    CompileTimeConstant,
    GenericMapping,
    Parameter,
    SpecializationItem,
    Type,
    TypeDefinition,
)
from .user_facing_errors import FailedLookupError, InvalidIntSize


class PlaceholderType(Type):
    class Definition(TypeDefinition):
        def __init__(self, generic_name: str) -> None:
            super().__init__()

            self.generic_name = generic_name

        def graphene_literal_to_ir_constant(self, value: str) -> str:
            assert False

        @cached_property
        def alignment(self) -> int:
            assert False

        @cached_property
        def size(self) -> int:
            assert False

        @cached_property
        def user_facing_name(self) -> str:
            return f"<placeholder for {self.generic_name}>"

        def get_ir_type(self, alias: Optional[str]) -> str:
            assert False

        @cached_property
        def ir_definition(self) -> str:
            assert False

        @cached_property
        def mangled_name(self) -> str:
            assert False

        @cached_property
        def is_void(self) -> bool:
            return False

        def __repr__(self) -> str:
            return f"PlaceholderType.Definition({self.generic_name})"

        def __eq__(self, other: Any) -> bool:
            assert False

    def __init__(self, generic_name: str) -> None:
        super().__init__(self.Definition(generic_name))

        self.generic_name = generic_name

    def match_with(self, other: Type, generic_mapping: GenericMapping) -> bool:
        # We are a placeholder, so we should just copy the other type.

        if (
            self.generic_name in generic_mapping
            and generic_mapping[self.generic_name] != other
        ):
            return False

        generic_mapping[self.generic_name] = other.copy()

        return True


@dataclass
class PlaceholderConstant(CompileTimeConstant):
    generic_name: str

    def match_with(
        self, other: "CompileTimeConstant", generic_mapping: GenericMapping
    ) -> bool:
        # TODO this is identical to PlaceholderType.match_with(), see if we can
        # merge the two definitions.
        if (
            self.generic_name in generic_mapping
            and generic_mapping[self.generic_name] != other
        ):
            return False

        generic_mapping[self.generic_name] = other

        return True


class PrimitiveDefinition(TypeDefinition):
    def __init__(self, size: int, ir_type: str, name: str) -> None:
        super().__init__()

        assert size > 0
        assert ir_type
        assert name

        self._size = size
        self._ir = ir_type
        self._name = name

    @cached_property
    def alignment(self) -> int:
        return self._size

    @cached_property
    def size(self) -> int:
        return self._size

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
        # Assert not reached: not all primitive types can be instantiated
        # anonymously. Those that can (e.g. structs, ints) overwrite this.
        assert False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        if isinstance(other, PrimitiveDefinition):
            # TODO: I'm not sure this is the correct approach
            return self._size == other._size and self._ir == other._ir

        return False


class IntegerDefinition(PrimitiveDefinition):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        super().__init__(size_in_bits // 8, f"i{size_in_bits}", name)

        self.is_signed = is_signed
        self.bits = size_in_bits

    def graphene_literal_to_ir_constant(self, value_str: str) -> str:
        if self.is_signed:
            range_lower = -(2 ** (self.bits - 1))
            range_upper = 2 ** (self.bits - 1)
            value = int(value_str)
        else:
            range_lower = 0
            range_upper = 2**self.bits
            value = int(value_str, 16)  # Unsigned ints are given in hex

        if not range_lower <= value < range_upper:
            raise InvalidIntSize(self._name, value, range_lower, range_upper)

        return str(value)

    @cached_property
    def mangled_name(self) -> str:
        return self._name

    def __eq__(self, other: Any) -> bool:
        if not super().__eq__(other):
            return False

        if isinstance(other, IntegerDefinition):
            return self.is_signed == other.is_signed and self.bits == other.bits

        return False


class VoidDefinition(TypeDefinition):
    def graphene_literal_to_ir_constant(self, value: str) -> str:
        assert False

    @cached_property
    def alignment(self) -> int:
        assert False

    @cached_property
    def size(self) -> int:
        assert False

    @cached_property
    def user_facing_name(self) -> str:
        return "void"

    def get_ir_type(self, alias: Optional[str]) -> str:
        return f"%alias.{alias}" if alias else self.ir_definition

    @cached_property
    def ir_definition(self) -> str:
        return "void"

    @cached_property
    def mangled_name(self) -> str:
        return "void"

    @cached_property
    def is_void(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "VoidDefinition"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, VoidDefinition)


class VoidType(Type):
    def __init__(self) -> None:
        super().__init__(VoidDefinition())


class GenericIntType(Type):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        is_power_of_2 = ((size_in_bits - 1) & size_in_bits) == 0
        is_divisible_into_bytes = (size_in_bits % 8) == 0
        assert is_power_of_2 and is_divisible_into_bytes

        definition = IntegerDefinition(name, size_in_bits, is_signed)
        super().__init__(definition)


class IntType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("int", 32, True)


class UIntType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("u32", 32, False)


class SizeType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("isize", 64, True)


class IPtrType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("iptr", 64, True)


class BoolType(Type):
    class Definition(PrimitiveDefinition):
        def __init__(self) -> None:
            super().__init__(1, "i1", "bool")

        def graphene_literal_to_ir_constant(self, value: str) -> str:
            # We happen to be using the same boolean constants as LLVM IR.
            assert value in ("true", "false")

            return value

    def __init__(self) -> None:
        definition = self.Definition()
        super().__init__(definition, definition._name)


class StructDefinition(TypeDefinition):
    def __init__(self, members: list[Parameter], is_adhoc: bool = False) -> None:
        super().__init__()

        # TODO: assert member names are unique
        self.members = members
        self.is_adhoc = is_adhoc

        # Different structs should compare different even if they have the same body
        self._uuid = uuid.uuid4()

    def get_member_by_name(self, name: str) -> tuple[int, Type]:
        for index, member in enumerate(self.members):
            if member.name == name:
                return index, member.type
        raise FailedLookupError("struct member", f"{{{name}: ...}}")

    def graphene_literal_to_ir_constant(self, value: str) -> str:
        # TODO support structure constants.
        raise NotImplementedError()

    @cached_property
    def user_facing_name(self) -> str:
        members = (
            f"{m.name} : {m.type.get_user_facing_name(False)}" for m in self.members
        )
        return f"{{{', '.join(members)}}}"

    def get_ir_type(self, alias: Optional[str]) -> str:
        # If this type has an alias, then we've generated a definition for it at
        # the top of the IR source.
        if alias:
            return f"%struct.{alias}"

        # If it doesn't, then we need to define the type here.
        return self.ir_definition

    @cached_property
    def ir_definition(self) -> str:
        member_ir = map(lambda m: m.type.ir_type, self.members)
        return "{" + str.join(", ", member_ir) + "}"

    @cached_property
    def mangled_name(self) -> str:
        subtypes = map(lambda m: m.type.mangled_name, self.members)
        return f"__ST{''.join(subtypes)}__TS"

    @cached_property
    def alignment(self) -> int:
        # TODO: can we be less conservative here
        return (
            max(member.type.alignment for member in self.members) if self.members else 1
        )

    @cached_property
    def size(self) -> int:
        # Return this size of the struct with c-style padding (for now)
        #   TODO: we should manually set the `datalayout` string to match this
        this_size = 0
        for member in self.members:
            # Add padding to ensure each member is aligned
            remainder = this_size % member.type.alignment
            this_size += (member.type.alignment - remainder) % member.type.alignment

            # Append member to the struct
            this_size += member.type.size

        # Add padding to align the final struct
        remainder = this_size % self.alignment
        this_size += (self.alignment - remainder) % self.alignment

        return this_size

    @cached_property
    def is_struct(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"StructDefinition({', '.join(map(repr, self.members))})"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        if not isinstance(other, StructDefinition):
            return False
        return self._uuid == other._uuid

    def match_with(
        self, other: TypeDefinition, generic_mapping: GenericMapping
    ) -> bool:
        if not isinstance(other, StructDefinition):
            return False

        if not self.is_adhoc and not other.is_adhoc:
            # UUID equality.
            return self == other

        # If at least one of the two structs are adhoc, we need to do pattern
        # matching.
        if len(self.members) != len(other.members):
            return False

        return all(
            map(
                lambda ours, theirs: ours.type.match_with(theirs.type, generic_mapping),
                self.members,
                other.members,
            )
        )


class ArrayDefinition(TypeDefinition):
    UNKNOWN_DIMENSION = CompileTimeConstant(0)

    def __init__(
        self, element_type: Type, dimensions: list[CompileTimeConstant]
    ) -> None:
        super().__init__()

        self._element_type = element_type
        self._dimensions = dimensions

    @cached_property
    def dimensions(self) -> list[CompileTimeConstant]:
        return self._dimensions

    def graphene_literal_to_ir_constant(self, value: str) -> str:
        # TODO support array constants.
        raise NotImplementedError()

    @cached_property
    def user_facing_name(self) -> str:
        type_name = self._element_type.get_user_facing_name(False)
        # TODO: ideally we would print T[&, 2, 3] for heap arrays
        #       atm `Type` would insert an extra (incorrect) reference symbol if we tried this
        dimensions = ", ".join(map(str, self._dimensions))
        return f"{type_name}[{dimensions}]"

    def get_ir_type(self, alias: Optional[str]) -> str:
        if alias:
            return f"%array.{alias}"

        return self.ir_definition

    @cached_property
    def ir_definition(self) -> str:
        def ir_sub_definition(dims: list[CompileTimeConstant]):
            if len(dims) == 0:
                return self._element_type.ir_type
            return f"[{dims[0].value} x {ir_sub_definition(dims[1:])}]"

        return ir_sub_definition(self._dimensions)

    @cached_property
    def mangled_name(self) -> str:
        dimensions = "_".join(map(str, self._dimensions))
        return f"__AR{self._element_type.mangled_name}_{dimensions}__AR"

    @cached_property
    def alignment(self) -> int:
        return self._element_type.alignment

    @cached_property
    def size(self) -> int:
        assert self._dimensions[0] != self.UNKNOWN_DIMENSION
        return self._element_type.size * reduce(
            mul, map(lambda d: d.value, self._dimensions), 1
        )

    def __repr__(self) -> str:
        dimensions = ", ".join(map(str, self._dimensions))
        return f"Array({self._element_type} x ({dimensions}))"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, TypeDefinition)
        if not isinstance(other, ArrayDefinition):
            return False
        return (
            self._dimensions == other._dimensions
            and self._element_type == other._element_type
        )

    def match_with(
        self, other: TypeDefinition, generic_mapping: GenericMapping
    ) -> bool:
        if not isinstance(other, ArrayDefinition):
            return False

        if not self._element_type.match_with(other._element_type, generic_mapping):
            return False

        return all(
            map(
                lambda l, r: l.match_with(r, generic_mapping),
                self.dimensions,
                other.dimensions,
            )
        )


class CharArrayDefinition(ArrayDefinition):
    def __init__(self, encoded_str: str, length: int) -> None:
        self._encoded_str = encoded_str
        super().__init__(GenericIntType("u8", 8, False), [CompileTimeConstant(length)])

    def graphene_literal_to_ir_constant(self, value: str) -> str:
        assert self._encoded_str == value
        return f'c"{value}"'


@dataclass
class FunctionSignature:
    name: str
    arguments: list[Type]
    return_type: Type
    specialization: list[SpecializationItem]
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

        arguments_mangle = "".join(arg.mangled_name for arg in self.arguments)

        specialization_mangle = ""
        if len(self.specialization) != 0:
            specialization_mangle = (
                "__SPECIAL_"
                + "".join(
                    specialization_item.mangled_name
                    if isinstance(specialization_item, Type)
                    else str(specialization_item)
                    for specialization_item in self.specialization
                )
                + "_SPECIAL__"
            )

        # Name mangle operators into digits
        legal_name_mangle = []
        for char in self.name:
            if char.isalnum() or char == "_":
                legal_name_mangle.append(char)
            else:
                legal_name_mangle.append(f"__O{ord(char)}")

        return f"{''.join(legal_name_mangle)}{specialization_mangle}{arguments_mangle}"

    def _repr_impl(self, key: Callable[[SpecializationItem], str]) -> str:
        prefix = "foreign" if self.is_foreign() else "function"

        readable_arg_names = ", ".join(map(key, self.arguments))

        if self.specialization:
            readable_specialization_types = ", ".join(map(key, self.specialization))
            fn_name = f"{self.name}<{readable_specialization_types}>"
        else:
            fn_name = self.name

        return (
            f"{prefix} {fn_name}: ({readable_arg_names}) -> " f"{key(self.return_type)}"
        )

    def __repr__(self) -> str:
        return self._repr_impl(repr)

    @cached_property
    def user_facing_name(self) -> str:
        return self._repr_impl(
            lambda t: t.get_user_facing_name(False) if isinstance(t, Type) else str(t)
        )

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.return_type.ir_type} @{self.mangled_name}"


def get_builtin_types() -> list[Type]:
    # Generate sized int types
    sized_int_types = []
    for size in (8, 16, 32, 64, 128):
        sized_int_types.append(GenericIntType(f"i{size}", size, True))
        sized_int_types.append(GenericIntType(f"u{size}", size, False))

    return sized_int_types + [
        BoolType(),
        IntType(),
        IPtrType(),
        SizeType(),
        VoidType(),
    ]

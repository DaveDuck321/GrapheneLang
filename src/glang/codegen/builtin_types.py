import uuid
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import mul
from typing import Callable, Optional

from ..target import get_abi, get_int_type_info, get_ptr_type_info
from .interfaces import SpecializationItem, Type, TypeDefinition, format_specialization
from .user_facing_errors import (
    ArrayDimensionError,
    FailedLookupError,
    InvalidIntSize,
    VoidArrayDeclaration,
    VoidStructDeclaration,
)


class PlaceholderDefinition(TypeDefinition):
    def are_equivalent(self, other: TypeDefinition) -> bool:
        assert other is self
        return True

    def format_for_output_to_user(self) -> str:
        assert False

    @property
    def is_finite(self) -> bool:
        # We can't verify if the type is finite right now because we haven't
        # substituted in all placeholder values, this is fine though because
        # the types we cyclically depend on will check us again when they've
        # actually had their placeholders substituted.
        return True

    @property
    def ir_mangle(self) -> str:
        assert False

    @property
    def ir_type(self) -> str:
        assert False

    @property
    def size(self) -> int:
        assert False

    @property
    def alignment(self) -> int:
        assert False


class NamedType(Type):
    def __init__(
        self,
        name: str,
        specialization: list[SpecializationItem],
    ) -> None:
        # NameType is initially created WITHOUT a definition
        # The definition is substituted later to allow recursive types
        super().__init__(PlaceholderDefinition(), False)

        self.name = name
        self.specialization = specialization
        self.alias: Optional[Type] = None

    def should_defer_to_alias_for_ir(self) -> bool:
        return isinstance(self.alias, NamedType)

    def update_with_finalized_alias(self, alias: Type) -> None:
        assert self.alias is None

        self.alias = alias
        self.definition = alias.definition
        self.is_reference = alias.is_reference

    def get_name(self) -> str:
        specialization = format_specialization(self.specialization)
        return f"{self.name}{specialization}"

    def format_for_output_to_user(self, full=False) -> str:
        reference_suffix = (
            "&"
            if self.is_reference and not self.definition.is_always_a_reference
            else ""
        )

        if not full:
            return self.get_name() + reference_suffix

        return (
            f"typedef {self.get_name()} : "
            f"{self.definition.format_for_output_to_user()}{reference_suffix}"
        )

    @property
    def ir_mangle(self) -> str:
        if self.should_defer_to_alias_for_ir():
            assert self.alias is not None
            return self.alias.ir_mangle

        prefix = "__R_" if self.is_reference else "__T_"

        if len(self.specialization) == 0:
            return prefix + self.name

        specializations = []
        for argument in self.specialization:
            if isinstance(argument, int):
                # TODO: negative numbers
                specializations.append(str(argument))
            elif isinstance(argument, Type):
                specializations.append(argument.ir_mangle)
            else:
                assert False

        specialization_format = "".join(specializations)
        return f"{prefix}{self.name}__S{specialization_format}"

    @property
    def ir_type(self) -> str:
        if self.is_reference:
            return "ptr"

        if self.should_defer_to_alias_for_ir():
            assert self.alias is not None
            return self.alias.ir_type

        return f"%type.{self.ir_mangle}"


class AnonymousType(Type):
    def __init__(self, definition: TypeDefinition) -> None:
        super().__init__(definition, definition.is_always_a_reference)

    def format_for_output_to_user(self, full=False) -> str:
        reference_suffix = (
            "&"
            if self.is_reference and not self.definition.is_always_a_reference
            else ""
        )
        return self.definition.format_for_output_to_user() + reference_suffix

    @property
    def ir_mangle(self) -> str:
        prefix = "__R_" if self.is_reference else "__ANON_"
        return prefix + self.definition.ir_mangle

    @property
    def ir_type(self) -> str:
        if self.is_reference:
            return "ptr"
        return self.definition.ir_type


class PrimitiveDefinition(TypeDefinition):
    def __init__(self, size: int, ir_type: str, name: str) -> None:
        super().__init__()

        assert size > 0
        assert ir_type
        assert name

        self._size = size
        self._ir = ir_type
        self._name = name

    def format_for_output_to_user(self) -> str:
        return self._name

    @property
    def is_finite(self) -> bool:
        return True

    @property
    def ir_type(self) -> str:
        return self._ir

    @property
    def ir_mangle(self) -> str:
        return self._ir

    @property
    def size(self) -> int:
        return self._size

    @property
    def alignment(self) -> int:
        return self._size


class PrimitiveType(NamedType):
    def __init__(self, name: str, definition: TypeDefinition) -> None:
        super().__init__(name, [])
        self.definition = definition

    def format_for_output_to_user(self, full=False) -> str:
        return super().format_for_output_to_user(False)

    @property
    def ir_type(self) -> str:
        if self.is_reference:
            return "ptr"
        return self.definition.ir_type


class VoidDefinition(PrimitiveDefinition):
    def __init__(self) -> None:
        super().__init__(1, "void", "void")

    def are_equivalent(self, other: TypeDefinition) -> bool:
        return isinstance(other, VoidDefinition)

    @property
    def size(self) -> int:
        assert False

    @property
    def alignment(self) -> int:
        assert False

    @property
    def is_void(self) -> bool:
        return True


class VoidType(PrimitiveType):
    def __init__(self) -> None:
        super().__init__("void", VoidDefinition())


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

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, IntegerDefinition):
            return False
        return self.is_signed == other.is_signed and self.bits == other.bits


class GenericIntType(PrimitiveType):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        is_power_of_2 = ((size_in_bits - 1) & size_in_bits) == 0
        is_divisible_into_bytes = (size_in_bits % 8) == 0
        assert is_power_of_2 and is_divisible_into_bytes

        definition = IntegerDefinition(name, size_in_bits, is_signed)
        super().__init__(name, definition)


class IntType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("int", get_int_type_info().size.in_bits, True)


class UIntType(GenericIntType):
    def __init__(self) -> None:
        super().__init__(
            f"u{get_int_type_info().size.in_bits}",
            get_int_type_info().size.in_bits,
            False,
        )


class SizeType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("isize", get_ptr_type_info().size.in_bits, True)


class IPtrType(GenericIntType):
    def __init__(self) -> None:
        super().__init__("iptr", get_ptr_type_info().size.in_bits, True)


class BoolDefinition(PrimitiveDefinition):
    def __init__(self) -> None:
        super().__init__(1, "i1", "bool")

    def are_equivalent(self, other: TypeDefinition) -> bool:
        return isinstance(other, BoolDefinition)

    def graphene_literal_to_ir_constant(self, value: str) -> str:
        # We happen to be using the same boolean constants as LLVM IR.
        assert value in ("true", "false")
        return value


class BoolType(PrimitiveType):
    def __init__(self) -> None:
        super().__init__("bool", BoolDefinition())


class StructDefinition(TypeDefinition):
    def __init__(self, members: list[tuple[str, Type]]) -> None:
        super().__init__()

        # The caller is responsible for ensuring that member names are unique.
        self.members = members
        self._uuid = uuid.uuid4()

        for member_name, member_type in self.members:
            if member_type.definition.is_void:
                raise VoidStructDeclaration(
                    self.format_for_output_to_user(),
                    member_name,
                    member_type.format_for_output_to_user(True),
                )

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, StructDefinition):
            return False

        return other._uuid == self._uuid

    def format_for_output_to_user(self) -> str:
        members_format = ", ".join(
            f"{member_name}: {member_type.format_for_output_to_user()}"
            for member_name, member_type in self.members
        )
        return "{" + members_format + "}"

    @property
    def is_finite(self) -> bool:
        for _, member_type in self.members:
            if not member_type.is_finite:
                return False
        return True

    @property
    def ir_type(self) -> str:
        member_ir = ", ".join(member.ir_type for _, member in self.members)
        return "{" + member_ir + "}"

    @property
    def ir_mangle(self) -> str:
        member_mangle = "".join(member.ir_mangle for _, member in self.members)
        return f"__S{member_mangle}"

    @property
    def size(self) -> int:
        member_sizes = [member.size for _, member in self.members]
        member_aligns = [member.alignment for _, member in self.members]
        return get_abi().compute_struct_size(member_sizes, member_aligns)

    @property
    def alignment(self) -> int:
        member_aligns = [member.alignment for _, member in self.members]
        return get_abi().compute_struct_alignment(member_aligns)

    def get_member_by_name(self, target_name: str) -> tuple[int, Type]:
        for index, (member_name, member_type) in enumerate(self.members):
            if member_name == target_name:
                return index, member_type

        raise FailedLookupError("struct member", "{" + target_name + " : ... }")


class StackArrayDefinition(TypeDefinition):
    def __init__(self, member: Type, dimensions: list[int]) -> None:
        super().__init__()

        self.member = member
        self.dimensions = dimensions

        if any(dim < 0 for dim in self.dimensions):
            raise ArrayDimensionError(self.format_for_output_to_user())

        if self.member.definition.is_void:
            raise VoidArrayDeclaration(
                self.format_for_output_to_user(),
                self.member.format_for_output_to_user(True),
            )

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, StackArrayDefinition):
            return False

        return other.member == self.member and other.dimensions == self.dimensions

    def format_for_output_to_user(self) -> str:
        dimensions_format = ", ".join(map(str, self.dimensions))
        return f"{self.member.format_for_output_to_user()}[{dimensions_format}]"

    @property
    def is_finite(self) -> bool:
        return self.member.is_finite

    @property
    def ir_type(self) -> str:
        return format_array_dims_for_ir(self.dimensions, self.member)

    @property
    def ir_mangle(self) -> str:
        dimensions = "_".join(map(str, self.dimensions))
        return f"__A_{self.member.ir_mangle}__{dimensions}"

    @property
    def size(self) -> int:
        return self.member.size * reduce(mul, self.dimensions, 1)

    @property
    def alignment(self) -> int:
        return get_abi().compute_stack_array_alignment(self.size, self.member.alignment)


class HeapArrayDefinition(TypeDefinition):
    def __init__(self, member: Type, known_dimensions: list[int]) -> None:
        super().__init__()

        self.member = member
        self.known_dimensions = known_dimensions

        if any(dim < 0 for dim in self.known_dimensions):
            raise ArrayDimensionError(self.format_for_output_to_user())

        if self.member.definition.is_void:
            raise VoidArrayDeclaration(
                self.format_for_output_to_user(),
                self.member.format_for_output_to_user(True),
            )

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, HeapArrayDefinition):
            return False

        return (
            other.member == self.member
            and other.known_dimensions == self.known_dimensions
        )

    def format_for_output_to_user(self) -> str:
        if len(self.known_dimensions) == 0:
            return f"{self.member.format_for_output_to_user()}[&]"

        dimensions_format = ", ".join(map(str, self.known_dimensions))
        return f"{self.member.format_for_output_to_user()}[&, {dimensions_format}]"

    @property
    def is_finite(self) -> bool:
        return False

    @property
    def ir_type(self) -> str:
        assert False  # The containing type should always be a reference

    @property
    def ir_mangle(self) -> str:
        dimensions = "_".join(map(str, self.known_dimensions))
        return f"__UA_{self.member.ir_mangle}__{dimensions}"

    @property
    def size(self) -> int:
        assert False

    @property
    def alignment(self) -> int:
        assert False

    @property
    def is_always_a_reference(self) -> bool:
        return True


class CharArrayDefinition(StackArrayDefinition):
    def __init__(self, encoded_str: str, length: int) -> None:
        self._encoded_str = encoded_str
        super().__init__(GenericIntType("u8", 8, False), [length])

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

    @property
    def is_main(self) -> bool:
        return self.name == "main"

    @property
    def is_foreign(self) -> bool:
        return self.foreign

    @cached_property
    def mangled_name(self) -> str:
        # main() is immune to name mangling (irrespective of arguments)
        if self.is_main or self.is_foreign:
            return self.name

        arguments_mangle = "".join(arg.ir_mangle for arg in self.arguments)

        specialization_mangle = ""
        if len(self.specialization) != 0:
            specialization_mangle = (
                "__SPECIAL_"
                + "".join(
                    specialization_item.ir_mangle
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
        prefix = "foreign" if self.is_foreign else "function"

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
            lambda t: t.format_for_output_to_user() if isinstance(t, Type) else str(t)
        )

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.return_type.ir_type} @{self.mangled_name}"

    def generate_declaration_ir(self) -> str:
        args_ir = ", ".join(arg.ir_type for arg in self.arguments)

        # XXX nounwind indicates that the function never raises an exception.
        return (
            f"declare dso_local {self.return_type.ir_type} "
            f"@{self.mangled_name}({args_ir}) nounwind"
        )


def get_builtin_types() -> list[PrimitiveType]:
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


def format_array_dims_for_ir(dims: list[int], element_type: Type) -> str:
    if not dims:
        return element_type.ir_type

    return f"[{dims[0]} x {format_array_dims_for_ir(dims[1:], element_type)}]"

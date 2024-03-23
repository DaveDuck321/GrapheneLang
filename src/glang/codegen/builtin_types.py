import uuid
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import mul

from glang.codegen.debug import (
    DIBasicType,
    DICompositeType,
    DIDerivedType,
    DISubrange,
    DIType,
    Metadata,
    MetadataList,
    Tag,
    TypeKind,
)
from glang.codegen.floats import float_literal_to_exact_hex
from glang.codegen.interfaces import (
    SpecializationItem,
    Type,
    TypeDefinition,
    format_specialization,
)
from glang.codegen.mangling import (
    mangle_float,
    mangle_int,
    mangle_operator_name,
    mangle_source_name,
    mangle_template_args,
)
from glang.codegen.user_facing_errors import (
    ArrayDimensionError,
    FailedLookupError,
    InvalidIntSize,
    VoidArrayDeclaration,
    VoidStructDeclaration,
)
from glang.target import get_abi, get_int_type_info, get_ptr_type_info


class PlaceholderDefinition(TypeDefinition):
    def are_equivalent(self, other: TypeDefinition) -> bool:
        assert other is self
        return True

    def format_for_output_to_user(self) -> str:
        raise AssertionError

    def copy_with_storage_kind(self, parent: Type, kind: Type.Kind) -> Type:
        assert kind.is_reference()
        return AnonymousType(ReferenceDefinition(parent, kind))

    @property
    def is_finite(self) -> bool:
        # We can't verify if the type is finite right now because we haven't
        # substituted in all placeholder values, this is fine though because
        # the types we cyclically depend on will check us again when they've
        # actually had their placeholders substituted.
        return True

    @property
    def ir_mangle(self) -> str:
        raise AssertionError

    @property
    def ir_type(self) -> str:
        raise AssertionError

    @property
    def size(self) -> int:
        raise AssertionError

    @property
    def alignment(self) -> int:
        raise AssertionError

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        raise AssertionError


class NamedType(Type):
    def __init__(
        self,
        name: str,
        specialization: list[SpecializationItem],
        definition: TypeDefinition,
        alias: Type | None,
    ) -> None:
        super().__init__(definition)

        self.name = name
        # TODO pass template parameter names; these are required for
        # DITemplateParameter.
        self.specialization = specialization
        self.alias: Type | None = alias

    def should_defer_to_alias_for_ir(self) -> bool:
        return isinstance(self.alias, NamedType)

    def update_with_finalized_alias(self, alias: Type) -> None:
        assert self.alias is None
        assert isinstance(self.definition, PlaceholderDefinition)

        self.alias = alias
        self.definition = alias.definition

    def get_name(self) -> str:
        return f"{self.name}{format_specialization(self.specialization)}"

    def format_for_output_to_user(self, full=False) -> str:
        name_fmt = self.get_name()
        if not full:
            return name_fmt

        return f"typedef {name_fmt} : {self.definition.format_for_output_to_user()}"

    def convert_to_storage_type(self, kind: Type.Kind) -> Type:
        if kind == self.storage_kind:
            return self

        if kind.is_reference():
            # Convert us (a value type) into a reference type
            return self.definition.copy_with_storage_kind(self, kind)

        # Convert us (a reference type) into a value type
        assert self.alias is not None
        return self.alias.convert_to_storage_type(kind)

    @property
    def ir_mangle(self) -> str:
        if self.should_defer_to_alias_for_ir():
            assert self.alias is not None
            return self.alias.ir_mangle

        # Perhaps built-in types shouldn't be a NamedType.
        if isinstance(self.definition, PrimitiveDefinition):
            return self.definition.ir_mangle

        source_name = mangle_source_name(self.name)
        if not self.specialization:
            return source_name

        return source_name + mangle_template_args(self.specialization)

    @property
    def ir_type(self) -> str:
        if self.should_defer_to_alias_for_ir():
            assert self.alias is not None
            return self.alias.ir_type

        return f"%type._Z{self.ir_mangle}"

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        metadata = super().to_di_type(metadata_gen)

        # The definition doesn't know its name.
        assert isinstance(metadata[-1], DIType)
        metadata[-1].name = self.get_name()

        return metadata


class AnonymousType(Type):
    def format_for_output_to_user(self, full=False) -> str:
        return self.definition.format_for_output_to_user()

    def convert_to_storage_type(self, kind: Type.Kind) -> Type:
        if kind == self.storage_kind:
            return self

        return self.definition.copy_with_storage_kind(self, kind)

    @property
    def ir_mangle(self) -> str:
        return self.definition.ir_mangle

    @property
    def ir_type(self) -> str:
        return self.definition.ir_type


class ValueTypeDefinition(TypeDefinition):
    def copy_with_storage_kind(self, parent: Type, kind: Type.Kind) -> Type:
        assert self.storage_kind != kind
        assert kind.is_reference()
        return AnonymousType(ReferenceDefinition(parent, kind))


class PtrTypeDefinition(TypeDefinition):
    def __init__(self, storage_kind: Type.Kind) -> None:
        super().__init__()
        # Illegal value types are allowed (used by arrays).
        self._storage_kind = storage_kind

    @abstractmethod
    def copy_with_storage_kind(self, _: Type, kind: Type.Kind) -> Type:
        pass

    @property
    def is_finite(self) -> bool:
        return True

    @property
    def ir_type(self) -> str:
        return "ptr"

    @property
    def size(self) -> int:
        return get_ptr_type_info().size.in_bytes

    @property
    def alignment(self) -> int:
        return get_ptr_type_info().align.in_bytes

    @property
    def storage_kind(self) -> Type.Kind:
        return self._storage_kind

    @property
    def is_mut(self) -> bool:
        return self.storage_kind.is_mutable_reference()


class PrimitiveDefinition(ValueTypeDefinition):
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
    def size(self) -> int:
        return self._size

    @property
    def alignment(self) -> int:
        return self._size


class PrimitiveType(NamedType):
    def __init__(self, name: str, definition: TypeDefinition) -> None:
        super().__init__(name, [], definition, None)

    def format_for_output_to_user(self, full=False) -> str:
        return super().format_for_output_to_user(False)

    @property
    def ir_type(self) -> str:
        return self.definition.ir_type


class VoidDefinition(PrimitiveDefinition):
    def __init__(self) -> None:
        super().__init__(1, "void", "void")

    def are_equivalent(self, other: TypeDefinition) -> bool:
        return isinstance(other, VoidDefinition)

    @property
    def size(self) -> int:
        raise AssertionError

    @property
    def alignment(self) -> int:
        raise AssertionError

    @property
    def is_void(self) -> bool:
        return True

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        raise AssertionError

    @property
    def ir_mangle(self) -> str:
        # <builtin-type> ::= v  # void
        return "v"


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

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        return [
            DIBasicType(
                next(metadata_gen),
                self._name,
                self.bits,
                Tag.base_type,
                TypeKind.signed if self.is_signed else TypeKind.unsigned,
            )
        ]

    @property
    def ir_mangle(self) -> str:
        return mangle_int(self.bits, is_signed=self.is_signed)


class GenericIntType(PrimitiveType):
    def __init__(self, name: str, size_in_bits: int, is_signed: bool) -> None:
        is_power_of_2 = ((size_in_bits - 1) & size_in_bits) == 0
        is_divisible_into_bytes = (size_in_bits % 8) == 0
        assert is_power_of_2
        assert is_divisible_into_bytes

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


class IEEEFloatDefinition(PrimitiveDefinition):
    # NOTE: we use IEEE rather than the target specific SIMD types
    def __init__(self, name: str, size_in_bits: int) -> None:
        ir_type = {16: "half", 32: "float", 64: "double", 128: "fp128"}[size_in_bits]
        super().__init__(size_in_bits // 8, ir_type, name)
        self.bits = size_in_bits

    def graphene_literal_to_ir_constant(self, value_str: str) -> str:
        return float_literal_to_exact_hex(value_str, 8 * self.size)

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, IEEEFloatDefinition):
            return False
        return self.size == other.size

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        return [
            DIBasicType(
                next(metadata_gen),
                self._name,
                self.bits,
                Tag.base_type,
                TypeKind.float,
            )
        ]

    @property
    def ir_mangle(self) -> str:
        return mangle_float(self.bits)


class IEEEFloat(PrimitiveType):
    def __init__(self, size_in_bits: int) -> None:
        name = f"f{size_in_bits}"
        super().__init__(name, IEEEFloatDefinition(name, size_in_bits))


class BoolDefinition(PrimitiveDefinition):
    def __init__(self) -> None:
        super().__init__(1, "i1", "bool")

    def are_equivalent(self, other: TypeDefinition) -> bool:
        return isinstance(other, BoolDefinition)

    def graphene_literal_to_ir_constant(self, value: str) -> str:
        # We happen to be using the same boolean constants as LLVM IR.
        assert value in ("true", "false")
        return value

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        return [
            DIBasicType(
                next(metadata_gen),
                self._name,
                self._size * 8,
                Tag.base_type,
                TypeKind.boolean,
            )
        ]

    @property
    def ir_mangle(self) -> str:
        # <builtin-type> ::= b  # bool
        return "b"


class BoolType(PrimitiveType):
    def __init__(self) -> None:
        super().__init__("bool", BoolDefinition())


class StructDefinition(ValueTypeDefinition):
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
        return all(member_type.is_finite for _, member_type in self.members)

    @property
    def ir_type(self) -> str:
        member_ir = ", ".join(member.ir_type for _, member in self.members)
        return "{" + member_ir + "}"

    @property
    def ir_mangle(self) -> str:
        member_types = (m_type for _, m_type in self.members)
        # <builtin-type> ::= u <source-name> [<template-args>]  # vendor extended type
        # FIXME binutils rejects the above :(.
        # `u <source-name>` works and so does `<source-name> <template-args>`,
        # so fall back to the latter until we can fix it (seems like a quick
        # fix in gcc/libiberty/cp-demangle.c).
        return (
            f"{mangle_source_name('AnonymousType')}{mangle_template_args(member_types)}"
        )

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

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        # TODO template parameters (see DWARF5 2.23).
        # HACK to get recursive types to work.
        if cached := getattr(self, "_di_type", None):
            assert isinstance(cached, DICompositeType)
            return [cached]

        self._di_type = DICompositeType(
            next(metadata_gen),
            None,
            8 * self.size,
            Tag.structure_type,
            None,
            None,
        )

        other_metadata: list[Metadata] = []
        di_derived_types: list[DIDerivedType] = []
        curr_offset = 0
        for m_name, m_type in self.members:
            base_type_metadata = m_type.to_di_type(metadata_gen)
            other_metadata.extend(base_type_metadata)
            assert isinstance(base_type_metadata[-1], DIType)

            di_derived_types.append(
                DIDerivedType(
                    next(metadata_gen),
                    m_name,
                    8 * m_type.size,
                    Tag.member,
                    base_type_metadata[-1],
                    8 * curr_offset,
                )
            )

            curr_offset += m_type.size

        metadata_list = MetadataList(next(metadata_gen), di_derived_types)
        self._di_type.elements = metadata_list

        return [*other_metadata, *di_derived_types, metadata_list, self._di_type]


class StackArrayDefinition(ValueTypeDefinition):
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
        #  <array-type> ::= A [<array bound number>] _ <element type>
        dimensions = str.join("", map(lambda d: f"A{d}_", self.dimensions))
        return dimensions + self.member.ir_mangle

    @property
    def size(self) -> int:
        return self.member.size * reduce(mul, self.dimensions, 1)

    @property
    def alignment(self) -> int:
        return get_abi().compute_stack_array_alignment(self.size, self.member.alignment)

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        base_type_metadata = self.member.to_di_type(metadata_gen)
        assert isinstance(base_type_metadata[-1], DIType)

        di_subranges = [DISubrange(next(metadata_gen), d) for d in self.dimensions]
        metadata_list = MetadataList(next(metadata_gen), di_subranges)

        return [
            *base_type_metadata,
            *di_subranges,
            metadata_list,
            DICompositeType(
                next(metadata_gen),
                None,
                8 * self.size,
                Tag.array_type,
                base_type_metadata[-1],
                metadata_list,
            ),
        ]


class ReferenceDefinition(PtrTypeDefinition):
    def __init__(self, value_type: Type, storage_kind: Type.Kind) -> None:
        assert storage_kind.is_reference()
        super().__init__(storage_kind)

        # TODO: user-facing error
        assert not value_type.storage_kind.is_reference()
        self.value_type = value_type

    def copy_with_storage_kind(self, parent: Type, kind: Type.Kind) -> Type:
        if kind == Type.Kind.VALUE:
            return self.value_type

        if kind == self.storage_kind:
            return parent

        assert self.storage_kind == Type.Kind.MUTABLE_OR_CONST_REF

        return AnonymousType(ReferenceDefinition(self.value_type, kind))

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, ReferenceDefinition):
            return False

        if other.is_mut != self.is_mut:
            return False

        return self.value_type == other.value_type

    def format_for_output_to_user(self) -> str:
        mut = " mut" if self.is_mut else ""
        return f"{self.value_type.format_for_output_to_user()}{mut}&"

    @property
    def ir_mangle(self) -> str:
        # <qualified-type>     ::= <qualifiers> <type>
        # <ref-qualifier>      ::= R                    # & ref-qualifier
        # <CV-qualifiers>      ::= [r] [V] [K]          # restrict (C99), volatile, const
        const = "" if self.is_mut else "K"
        return "R" + const + self.value_type.ir_mangle

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        base_type_metadata = self.value_type.to_di_type(metadata_gen)
        assert isinstance(base_type_metadata[-1], DIType)

        return [
            *base_type_metadata,
            DIDerivedType(
                next(metadata_gen),
                None,
                8 * self.size,
                Tag.reference_type,
                base_type_metadata[-1],
                None,
            ),
        ]


class HeapArrayDefinition(PtrTypeDefinition):
    def __init__(
        self, member: Type, known_dimensions: list[int], storage: Type.Kind
    ) -> None:
        super().__init__(storage)

        self.member = member
        self.known_dimensions = known_dimensions
        self.is_illegal_value_type = not storage.is_reference()

        if any(dim < 0 for dim in self.known_dimensions):
            raise ArrayDimensionError(self.format_for_output_to_user())

        if self.member.definition.is_void:
            raise VoidArrayDeclaration(
                self.format_for_output_to_user(),
                self.member.format_for_output_to_user(True),
            )

    def copy_with_storage_kind(self, _: Type, kind: Type.Kind) -> Type:
        return AnonymousType(
            HeapArrayDefinition(self.member, self.known_dimensions, kind)
        )

    def are_equivalent(self, other: TypeDefinition) -> bool:
        if not isinstance(other, HeapArrayDefinition):
            return False

        return (
            other.is_mut == self.is_mut
            and other.known_dimensions == self.known_dimensions
            and other.is_illegal_value_type == self.is_illegal_value_type
            and other.member == self.member
        )

    def format_for_output_to_user(self) -> str:
        if self.is_illegal_value_type:
            ref_str = "<unborrowed>"
        else:
            ref_str = "mut&" if self.is_mut else "&"

        if len(self.known_dimensions) == 0:
            return f"{self.member.format_for_output_to_user()}[{ref_str}]"

        dim_fmt = ", ".join(map(str, self.known_dimensions))
        return f"{self.member.format_for_output_to_user()}[{ref_str}, {dim_fmt}]"

    @property
    def ir_mangle(self) -> str:
        assert not self.is_illegal_value_type
        #  <array-type> ::= A [<array bound number>] _ <element type>
        known_dimensions = str.join("", map(lambda d: f"A{d}_", self.known_dimensions))
        const = "" if self.is_mut else "K"
        return const + "PA_" + known_dimensions + self.member.ir_mangle

    @property
    def storage_kind(self) -> Type.Kind:
        if self.is_illegal_value_type:
            return Type.Kind.VALUE
        return super().storage_kind

    def to_di_type(self, metadata_gen: Iterator[int]) -> list[Metadata]:
        base_type_metadata = self.member.to_di_type(metadata_gen)
        assert isinstance(base_type_metadata[-1], DIType)

        return [
            *base_type_metadata,
            DIDerivedType(
                next(metadata_gen),
                None,
                8 * self.size,
                Tag.pointer_type,
                base_type_metadata[-1],
                None,
            ),
        ]


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

        # <mangled-name> ::= _Z <encoding>
        # <encoding> ::= <function name> <bare-function-type>

        # <name> ::= <unscoped-name>
        #        ::= <unscoped-template-name> <template-args>
        # <unscoped-template-name> ::= <unscoped-name>
        # <unscoped-name> ::= <unqualified-name>
        # <unqualified-name> ::= <operator-name> [<abi-tags>]
        #                    ::= <source-name>

        # The above reduces to:
        # <name> ::= (<operator-name> | <source-name>) [<template-args>]

        mangled_name = (
            mangle_source_name(self.name)
            if self.name.replace("_", "").isalnum()
            else mangle_operator_name(self.name)
        )

        mangled_template_args = (
            mangle_template_args(self.specialization) if self.specialization else ""
        )

        # <bare-function-type> ::= <signature type>+
        # Types are possible return type, then parameter types.

        # Whether the mangling of a function type includes the return type
        # depends on the context and the nature of the function. The rules for
        # deciding whether the return type is included are:

        # (1) Template functions (names or types) have return types encoded,
        # with the exceptions listed below.

        # (2) Function types not appearing as part of a function name mangling,
        # e.g. parameters, pointer types, etc., have return type encoded, with
        # the exceptions listed below.

        # (3) Non-template function names do not have return types encoded.

        # The exceptions mentioned in (1) and (2) above, for which the return
        # type is never included, are
        # - Constructors.
        # - Destructors.
        # - Conversion operator functions, e.g. operator int.

        mangled_return_type = self.return_type.ir_mangle if self.specialization else ""
        mangled_argument_types = "".join(arg.ir_mangle for arg in self.arguments)
        if not mangled_argument_types:
            # Empty parameter lists, whether declared as () or conventionally as
            # (void), are encoded with a void parameter specifier (v). Therefore
            # function types always encode at least one parameter type, and
            # function manglings can always be distinguished from data manglings
            # by the presence of the type.
            mangled_argument_types = "v"

        return (
            "_Z"
            + mangled_name
            + mangled_template_args
            + mangled_return_type
            + mangled_argument_types
        )

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

    return [
        *sized_int_types,
        BoolType(),
        IntType(),
        IPtrType(),
        SizeType(),
        VoidType(),
        IEEEFloat(16),
        IEEEFloat(32),
        IEEEFloat(64),
        IEEEFloat(128),
    ]


def format_array_dims_for_ir(dims: list[int], element_type: Type) -> str:
    if not dims:
        return element_type.ir_type

    return f"[{dims[0]} x {format_array_dims_for_ir(dims[1:], element_type)}]"

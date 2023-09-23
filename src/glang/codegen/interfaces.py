from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Optional

from .user_facing_errors import MutableVariableContainsAReference


class TypeDefinition(ABC):
    def graphene_literal_to_ir_constant(self, value: str) -> str:
        assert False

    @abstractmethod
    def are_equivalent(self, other: "TypeDefinition") -> bool:
        pass

    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    @abstractmethod
    def copy_with_storage_kind(self, parent_type: "Type", kind: "Type.Kind") -> "Type":
        pass

    @property
    @abstractmethod
    def ir_mangle(self) -> str:
        pass

    @property
    @abstractmethod
    def ir_type(self) -> str:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def alignment(self) -> int:
        pass

    @property
    @abstractmethod
    def is_finite(self) -> bool:
        pass

    @property
    def is_void(self) -> bool:
        return False

    @property
    def storage_kind(self) -> "Type.Kind":
        return Type.Kind.VALUE

    def __repr__(self) -> str:
        return f"TypeDefinition({self.format_for_output_to_user()})"


class Type(ABC):
    class Kind(Enum):
        VALUE = 1
        MUTABLE_REF = 2
        CONST_REF = 3

        def is_reference(self) -> bool:
            return self != Type.Kind.VALUE

    def __init__(self, definition: TypeDefinition) -> None:
        self.definition = definition

        self._visited_in_finite_resolution = False

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Type)
        return self.definition.are_equivalent(other.definition)

    @property
    def is_finite(self) -> bool:
        if self._visited_in_finite_resolution:
            return False

        self._visited_in_finite_resolution = True
        is_finite = self.definition.is_finite
        self._visited_in_finite_resolution = False
        return is_finite

    @property
    def size(self) -> int:
        return self.definition.size

    @property
    def alignment(self) -> int:
        return self.definition.alignment

    @property
    def storage_kind(self) -> Kind:
        return self.definition.storage_kind

    def convert_to_value_type(self) -> "Type":
        return self.convert_to_storage_type(Type.Kind.VALUE)

    @abstractmethod
    def convert_to_storage_type(self, kind: Kind) -> "Type":
        pass

    @abstractmethod
    def format_for_output_to_user(self, full=False) -> str:
        pass

    @property
    @abstractmethod
    def ir_mangle(self) -> str:
        pass

    @property
    @abstractmethod
    def ir_type(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"Type({self.format_for_output_to_user(True)})"


class Variable(ABC):
    def __init__(self, name: str, var_type: Type, is_mutable: bool) -> None:
        super().__init__()

        self._name = name
        self.type = var_type
        self.is_mutable = is_mutable

        # We cannot store references in mutable variables. Since there is no
        #  syntax to reassign the reference
        if self.type.storage_kind.is_reference() and self.is_mutable:
            raise MutableVariableContainsAReference(
                self._name, self.type.format_for_output_to_user(True)
            )

        self.ir_reg: Optional[int] = None

    @property
    def user_facing_name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def ir_ref_without_type_annotation(self) -> str:
        pass

    @property
    @abstractmethod
    def ir_ref(self) -> str:
        pass

    @abstractmethod
    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name}: {repr(self.type)}, is_mut: {self.is_mutable})"

    @abstractmethod
    def assert_can_read_from(self) -> None:
        pass

    @abstractmethod
    def assert_can_write_to(self) -> None:
        pass


class Generatable(ABC):
    def generate_ir(self, _: Iterator[int]) -> list[str]:
        return []

    @abstractmethod
    def is_return_guaranteed(self) -> bool:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @staticmethod
    def expand_ir(
        generatables: Iterable["Generatable"], reg_gen: Iterator[int]
    ) -> list[str]:
        ir_lines: list[str] = []

        for generatable in generatables:
            ir_lines.extend(generatable.generate_ir(reg_gen))

        return ir_lines


class TypedExpression(Generatable):
    def __init__(
        self,
        underlying_indirection_kind: Type.Kind,
    ) -> None:
        super().__init__()
        self.underlying_indirection_kind = underlying_indirection_kind
        self.result_reg: Optional[int] = None

    @property
    @abstractmethod
    def result_type(self) -> Type:
        pass

    @property
    def result_type_as_if_borrowed(self) -> Type:
        if self.underlying_indirection_kind.is_reference():
            return self.result_type.convert_to_storage_type(
                self.underlying_indirection_kind
            )
        return self.result_type

    @property
    @abstractmethod
    def has_address(self) -> bool:
        pass

    def is_return_guaranteed(self) -> bool:
        # At the moment no TypedExpression can return
        return False

    @property
    def ir_ref_with_type_annotation(self) -> str:
        assert self.ir_ref_without_type_annotation is not None
        return f"{self.ir_type_annotation} {self.ir_ref_without_type_annotation}"

    @property
    @abstractmethod
    def ir_type_annotation(self) -> str:
        pass

    def dereference_double_indirection(
        self, reg_gen: Iterator[int], ir: list[str]
    ) -> int:
        # Converts a double indirection eg. address of reference into a reference
        assert self.has_address

        store_at = next(reg_gen)
        ir.append(
            f"%{store_at} = load ptr, {self.ir_ref_with_type_annotation}, "
            f"align {self.result_type_as_if_borrowed.alignment}"
        )
        return store_at

    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    @abstractmethod
    def try_convert_to_type(self, type: Type) -> tuple[int, list["TypedExpression"]]:
        pass

    @property
    @abstractmethod
    def ir_ref_without_type_annotation(self) -> str:
        pass

    @abstractmethod
    def assert_can_read_from(self) -> None:
        pass

    @abstractmethod
    def assert_can_write_to(self) -> None:
        pass


class StaticTypedExpression(TypedExpression):
    def __init__(
        self,
        expr_type: Type,
        underlying_indirection_kind: Type.Kind,
        was_reference_type_at_any_point: bool = False,
    ) -> None:
        # It is the caller's responsibility to escape double indirections
        if expr_type.storage_kind.is_reference():
            assert not underlying_indirection_kind.is_reference()

        self.underlying_type = expr_type

        self.was_reference_type_at_any_point = was_reference_type_at_any_point
        super().__init__(underlying_indirection_kind)

    @property
    def result_type(self) -> Type:
        return self.underlying_type

    @property
    def has_address(self) -> bool:
        return (
            self.underlying_type.storage_kind.is_reference()
            or self.underlying_indirection_kind.is_reference()
        )

    def format_for_output_to_user(self) -> str:
        return self.underlying_type.format_for_output_to_user()

    def try_convert_to_type(self, type: Type) -> tuple[int, list[TypedExpression]]:
        return (0, [])

    @property
    def ir_type_annotation(self) -> str:
        if self.underlying_indirection_kind.is_reference():
            return "ptr"

        return self.underlying_type.ir_type


SpecializationItem = Type | int


@dataclass(frozen=True)
class GenericArgument:
    name: str
    is_value_arg: bool


@dataclass
class GenericMapping:
    mapping: dict[GenericArgument, SpecializationItem]
    pack: list[Type]


def do_specializations_match(
    s1: list[SpecializationItem], s2: list[SpecializationItem]
) -> bool:
    if len(s1) != len(s2):
        return False

    for item1, item2 in zip(s1, s2):
        if isinstance(item1, Type) != isinstance(item2, Type):
            return False

        if item1 != item2:
            return False

    return True


def format_specialization(specialization: list[SpecializationItem]) -> str:
    if len(specialization) == 0:
        return ""

    items = ", ".join(
        item.format_for_output_to_user() if isinstance(item, Type) else str(item)
        for item in specialization
    )
    return f"<{items}>"


def format_arguments(args: Iterable[Type] | Iterable[TypedExpression]) -> str:
    items = ", ".join(item.format_for_output_to_user() for item in args)
    return f"({items})"


def format_generics(args: Iterable[GenericArgument], pack_name: Optional[str]) -> str:
    formatted_generics = [item.name for item in args]
    if pack_name is not None:
        formatted_generics.append(pack_name)

    if len(formatted_generics) == 0:
        return ""

    return f" [{str.join(', ', formatted_generics)}]"

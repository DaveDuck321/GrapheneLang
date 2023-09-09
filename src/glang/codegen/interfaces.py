from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

from ..target import get_llvm_type_info


class TypeDefinition(ABC):
    def graphene_literal_to_ir_constant(self, value: str) -> str:
        assert False

    @abstractmethod
    def are_equivalent(self, other: "TypeDefinition") -> bool:
        pass

    @abstractmethod
    def format_for_output_to_user(self) -> str:
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
    def is_always_a_reference(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"TypeDefinition({self.format_for_output_to_user()})"


class Type(ABC):
    def __init__(self, definition: TypeDefinition, is_reference: bool) -> None:
        self.definition = definition
        self.is_reference = is_reference

        self._visited_in_finite_resolution = False

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Type)
        return (
            other.is_reference == self.is_reference
            and self.definition.are_equivalent(other.definition)
        )

    @property
    def is_finite(self) -> bool:
        if self.is_reference:
            return True

        if self._visited_in_finite_resolution:
            return False

        self._visited_in_finite_resolution = True
        is_finite = self.definition.is_finite
        self._visited_in_finite_resolution = False
        return is_finite

    @property
    def size(self) -> int:
        if self.is_reference:
            return get_llvm_type_info("ptr").size.in_bytes
        return self.definition.size

    @property
    def alignment(self) -> int:
        if self.is_reference:
            return get_llvm_type_info("ptr").align.in_bytes
        return self.definition.alignment

    def convert_to_value_type(self) -> "Type":
        result = copy(self)
        result.is_reference = False
        return result

    def convert_to_reference_type(self) -> "Type":
        result = copy(self)
        result.is_reference = True
        return result

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
        return f"Type({self.format_for_output_to_user()})"


class Variable(ABC):
    # TODO much of this interface is common with TypedExpression. Maybe they
    # should have a shared base class.
    def __init__(self, name: str, var_type: Type, constant: bool) -> None:
        super().__init__()

        self._name = name
        self.type = var_type
        self.constant = constant

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
        return f"{self.__class__.__name__}({self._name}: {repr(self.type)})"

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
        expr_type: Optional[Type],
        is_indirect_pointer_to_type: bool,
        was_reference_type_at_any_point: bool = False,
    ) -> None:
        super().__init__()
        # It is the callers responsibility to escape double indirections
        if expr_type is not None and expr_type.is_reference:
            assert not is_indirect_pointer_to_type

        self._underlying_type = expr_type
        self.is_indirect_pointer_to_type = is_indirect_pointer_to_type

        # Used for better error messages
        self.was_reference_type_at_any_point = was_reference_type_at_any_point

        self.result_reg: Optional[int] = None

    @property
    def underlying_type(self) -> Type:
        assert self._underlying_type is not None
        return self._underlying_type

    def get_equivalent_pure_type(self) -> Type:
        if self.is_indirect_pointer_to_type:
            return self.underlying_type.convert_to_reference_type()
        return self.underlying_type

    @property
    def has_address(self) -> bool:
        return self.underlying_type.is_reference or self.is_indirect_pointer_to_type

    def is_return_guaranteed(self) -> bool:
        # At the moment no TypedExpression can return
        return False

    @property
    def ir_ref_with_type_annotation(self) -> str:
        assert self.ir_ref_without_type_annotation is not None
        return f"{self.ir_type_annotation} {self.ir_ref_without_type_annotation}"

    @property
    def ir_type_annotation(self) -> str:
        if self.is_indirect_pointer_to_type:
            return "ptr"

        return self.underlying_type.ir_type

    def dereference_double_indirection(
        self, reg_gen: Iterator[int], ir: list[str]
    ) -> int:
        # Converts a double indirection eg. address of reference into a reference
        assert self.has_address

        store_at = next(reg_gen)
        ir.append(
            f"%{store_at} = load ptr, {self.ir_ref_with_type_annotation}, "
            f"align {self.get_equivalent_pure_type().alignment}"
        )
        return store_at

    def format_for_output_to_user(self) -> str:
        return self.underlying_type.format_for_output_to_user()

    def try_convert_to_type(self, type: Type) -> tuple[int, list["TypedExpression"]]:
        return (0, [])

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


@dataclass
class Parameter:
    name: str
    type: Type

    def __eq__(self, _: Any) -> bool:
        # No one was using this :).
        assert False

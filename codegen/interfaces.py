from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Iterator, Optional


class TypeDefinition(ABC):
    @abstractmethod
    def to_ir_constant(self, value: str) -> str:
        pass

    @abstractmethod
    def get_alignment(self) -> int:
        pass

    @cached_property
    @abstractmethod
    def mangled_name_for_ir(self) -> str:
        pass

    @abstractmethod
    def get_anonymous_ir_type_def(self) -> str:
        pass

    @abstractmethod
    def get_named_ir_type_ref(self, name: str) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class Type:
    is_reference = False

    def __init__(self, definition: TypeDefinition, name: Optional[str] = None) -> None:
        self.definition = definition

        if name is None:
            self.user_facing_typedef_assigned_name = "__anonymous"
            self.is_anonymous = True
        else:
            self.user_facing_typedef_assigned_name = name
            self.is_anonymous = False

    def __repr__(self) -> str:
        if self.is_anonymous:
            return f"Type({repr(self.definition)})"
        return f"Type({self.user_facing_typedef_assigned_name})"

    def __eq__(self, other: Any) -> bool:
        assert isinstance(self, Type)
        assert isinstance(other, Type)

        return self.definition == other.definition

    def get_alignment(self) -> int:
        return self.definition.get_alignment()

    @cached_property
    def ir_type_annotation(self) -> str:
        if self.is_anonymous:
            return self.definition.get_anonymous_ir_type_def()
        else:
            return self.definition.get_named_ir_type_ref(self.mangled_name_for_ir)

    def get_ir_initial_type_def(self) -> list[str]:
        assert not self.is_anonymous
        named_ref = self.definition.get_named_ir_type_ref(self.mangled_name_for_ir)
        definition = self.definition.get_anonymous_ir_type_def()
        return [f"{named_ref} = type {definition}"]

    def get_non_reference_type(self) -> "Type":
        return self

    @cached_property
    def mangled_name_for_ir(self) -> str:
        if self.is_anonymous:
            return f"__T__anon_{self.definition.mangled_name_for_ir}"
        else:
            return f"__T_{self.user_facing_typedef_assigned_name}"


@dataclass
class Parameter:
    name: str
    type: Type

    def __eq__(self, _: Any) -> bool:
        # No one was using this :).
        assert False


class Variable(ABC):
    # TODO much of this interface is common with TypedExpression. Maybe they
    # should have a shared base class.
    def __init__(self, name: str, type: Type, constant: bool) -> None:
        super().__init__()

        self.user_facing_graphene_name = name
        self.type = type
        self.constant = constant

        self.ir_reg: Optional[int] = None

    @cached_property
    @abstractmethod
    def ir_ref_without_type_annotation(self) -> str:
        pass

    @cached_property
    @abstractmethod
    def ir_ref(self) -> str:
        pass

    @abstractmethod
    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        pass


class Generatable(ABC):
    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return []

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
    def __init__(self, type: Type) -> None:
        super().__init__()

        self.type = type
        self.result_reg: Optional[int] = None

    @cached_property
    def ir_ref_with_type_annotation(self) -> str:
        type_annotation = self.type.ir_type_annotation
        reference = self.ir_ref_without_type_annotation
        return f"{type_annotation} {reference}"

    @cached_property
    @abstractmethod
    def ir_ref_without_type_annotation(self) -> str:
        pass

    @abstractmethod
    def assert_can_read_from(self) -> None:
        pass

    @abstractmethod
    def assert_can_write_to(self) -> None:
        pass

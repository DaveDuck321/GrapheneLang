from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterator, Optional


class TypeDefinition(ABC):
    @abstractmethod
    def compatible_with(self, value: Any) -> bool:
        pass

    @abstractmethod
    def cast_constant(self, value: int) -> bool:
        pass

    @abstractmethod
    def get_alignment(self) -> int:
        pass

    @abstractmethod
    def get_anonymous_ir_ref(self) -> str:
        pass

    @abstractmethod
    def get_named_ir_ref(self, name: str) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class Type:
    is_reference = False

    def __init__(self, definition: TypeDefinition, name: str = "anonymous") -> None:
        self.definition = definition
        self.name = name

    def __repr__(self) -> str:
        if self.name == "anonymous":
            return repr(self.definition)
        return self.name

    def is_implicitly_convertible_to(self, other: "Type") -> bool:
        assert isinstance(other, Type)
        return self.definition == other.definition

    @cached_property
    def align(self) -> int:
        return self.definition.get_alignment()

    @cached_property
    def ir_type(self) -> str:
        if self.name == "anonymous":
            return self.definition.get_anonymous_ir_ref()
        else:
            return self.definition.get_named_ir_ref(self.name)

    def get_definition_ir(self) -> list[str]:
        if self.name == "anonymous":
            return []
        named_ref = self.definition.get_named_ir_ref(self.name)
        anonymous_definition = self.definition.get_anonymous_ir_ref()
        return [f"{named_ref} = type {anonymous_definition}"]

    @cached_property
    def mangled_name(self) -> str:
        return "__T_TODO_NAME_MANGLE_TYPE"

    def __eq__(self, other: Any) -> bool:
        assert False  # We can only compare TypeDefinitions


@dataclass
class Variable:
    name: str
    type: Type

    def __eq__(self, other: Any) -> bool:
        return self.name == other.name and self.type.is_implicitly_convertible_to(
            other.type
        )


class Generatable(ABC):
    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return []

    @abstractmethod
    def __repr__(self) -> str:
        pass


class TypedExpression(Generatable):
    def __init__(self, type: Type) -> None:
        super().__init__()

        self.type = type
        self.result_reg: Optional[int] = None

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} {self.ir_ref_without_type}"

    @cached_property
    @abstractmethod
    def ir_ref_without_type(self) -> str:
        pass

    @abstractmethod
    def assert_can_read_from(self) -> None:
        pass

    @abstractmethod
    def assert_can_write_to(self) -> None:
        pass

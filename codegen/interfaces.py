from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterator, Optional


class Type(ABC):
    align = 1  # Unaligned
    ir_type = ""
    is_reference = False

    def __init__(self, name, definition) -> None:
        self.name = name
        self.definition = definition

    def __repr__(self) -> str:
        return self.name

    @cached_property
    def mangled_name(self) -> str:
        return "__T_TODO_NAME_MANGLE_TYPE"

    @abstractmethod
    def compatible_with(self, value: Any) -> bool:
        pass

    @abstractmethod
    def cast_constant(self, value: int) -> bool:
        pass

    def __eq__(self, other: Any) -> bool:
        # TODO how do we do comparisons with reference types?
        assert isinstance(other, Type)
        return self.name == other.name and self.definition == other.definition


@dataclass
class Parameter:
    name: str
    type: Type


class Variable(ABC):
    def __init__(self, name: str, type: Type, constant: bool) -> None:
        super().__init__()

        self.name = name
        self.type = type
        self.constant = constant

        self.ir_reg: Optional[int] = None

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

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
        # FIXME should also be a property.
        pass

    @cached_property
    @abstractmethod
    def user_facing_name(self) -> str:
        pass

    @abstractmethod
    def get_ir_type(self, alias: Optional[str]) -> str:
        pass

    @cached_property
    @abstractmethod
    def ir_definition(self) -> str:
        pass

    @cached_property
    @abstractmethod
    def mangled_name(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class Type:
    is_reference = False

    @staticmethod
    def mangle_list(types: list["Type"]) -> str:
        generic_mangles = map(lambda t: t.mangled_name, types)
        return str.join("", generic_mangles)

    @staticmethod
    def mangle_generic_type(alias: str, generics: Optional[list["Type"]]) -> str:
        name = f"__T_{alias}"

        if generics:
            generic_mangle = Type.mangle_list(generics)
            name += f"__G_{generic_mangle}"

        return name

    def __init__(
        self,
        definition: TypeDefinition,
        typedef_alias: Optional[str] = None,
        generic_args: Optional[list["Type"]] = None,
    ) -> None:
        self.definition = definition
        self._typedef_alias = typedef_alias  # Name given in typedef, without generics.
        self._generic_args = generic_args

    @cached_property
    def generic_annotation(self) -> str:
        if not self._generic_args:
            return ""

        generic_names = map(
            lambda arg: arg.get_user_facing_name(True), self._generic_args
        )

        return f"[{', '.join(generic_names)}]"

    def __repr__(self) -> str:
        name = f"{self._typedef_alias} = " if self._typedef_alias else ""
        name += repr(self.definition)

        return f"{self.__class__.__name__}({name})"

    def get_user_facing_name(self, full: bool) -> str:
        # Return everything (that's available).
        if full:
            name = f"typedef {self._typedef_alias} = " if self._typedef_alias else ""
            name += self.definition.user_facing_name

            # FIXME returns e.g. "typedef int = int"  for primitive types.
            return name

        # If this is the product of a typedef, return the name given.
        if self._typedef_alias:
            return self._typedef_alias

        # Fall back to the type definition.
        return self.definition.user_facing_name

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Type)

        return self.definition == other.definition

    def get_alignment(self) -> int:
        return self.definition.get_alignment()

    @cached_property
    def ir_type(self) -> str:
        if self._typedef_alias:
            return self.definition.get_ir_type(self.mangled_name)

        return self.definition.get_ir_type(None)

    def get_ir_initial_type_def(self) -> list[str]:
        assert self._typedef_alias
        named_ref = self.definition.get_ir_type(self.mangled_name)
        definition = self.definition.ir_definition
        return [f"{named_ref} = type {definition}"]

    def get_non_reference_type(self) -> "Type":
        return self

    @cached_property
    def mangled_name(self) -> str:
        alias = self._typedef_alias or f"anon_{self.definition.mangled_name}"

        return self.mangle_generic_type(alias, self._generic_args)


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
    def __init__(self, name: str, var_type: Type, constant: bool) -> None:
        super().__init__()

        self.user_facing_graphene_name = name
        self.type = var_type
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
    def generate_ir(self, _: Iterator[int]) -> list[str]:
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
    def __init__(self, expr_type: Type) -> None:
        super().__init__()

        self.type = expr_type
        self.result_reg: Optional[int] = None

    @cached_property
    def ir_ref_with_type_annotation(self) -> str:
        type_annotation = self.type.ir_type
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

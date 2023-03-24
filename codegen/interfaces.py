from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Iterator, Optional


class TypeDefinition(ABC):
    @abstractmethod
    def to_ir_constant(self, value: str) -> str:
        pass

    @cached_property
    @abstractmethod
    def alignment(self) -> int:
        pass

    @cached_property
    @abstractmethod
    def size(self) -> int:
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

    @cached_property
    def is_void(self) -> bool:
        return False

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class Type:
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
        specialization: list["Type"] = [],
    ) -> None:
        self.definition = definition
        self._specialization = specialization

        # TODO explanation.
        self.is_unborrowed_ref: bool = False
        self.is_borrowed: bool = False
        self._ref_depth: int = 0

        # Name given in typedef, without generics.
        self._typedef_alias: Optional[str] = None
        # Reference depth when the alias was set.
        self._typedef_alias_ref_depth: int = 0

        if typedef_alias is not None:
            self._set_typedef_alias(typedef_alias)

    def _set_typedef_alias(self, typedef_alias: str) -> None:
        self._typedef_alias = typedef_alias
        self._typedef_alias_ref_depth = self._ref_depth

    def get_specialization(self) -> list["Type"]:
        return self._specialization.copy()

    @property
    def ref_depth(self) -> int:
        assert self._ref_depth >= 0

        return self._ref_depth

    @property
    def is_reference(self) -> bool:
        return self.ref_depth > 0

    @property
    def is_pointer(self) -> bool:
        return self.is_unborrowed_ref or self.is_reference

    @property
    def is_void(self) -> bool:
        return self.definition.is_void

    @property
    def generic_annotation(self) -> str:
        if not self._specialization:
            return ""

        generic_names = map(
            lambda arg: arg.get_user_facing_name(True), self._specialization
        )

        return f"<{', '.join(generic_names)}>"

    def __repr__(self) -> str:
        name = f"{self._typedef_alias} = " if self._typedef_alias else ""
        name += repr(self.definition)

        return (
            f"{self.__class__.__name__}({name}, "
            f"is_unborrowed_ref={self.is_unborrowed_ref}, "
            f"borrowed={self.is_borrowed}, "
            f"ref_depth={self._ref_depth})"
        )

    def get_user_facing_name(self, full: bool) -> str:
        prefix = "*" * max(self._typedef_alias_ref_depth - self._ref_depth, 0)
        suffix = self.generic_annotation + (
            # This works in all cases because self._typedef_alias_ref_depth
            # is 0 if self._typedef_alias is not set.
            "&" * max(self._ref_depth - self._typedef_alias_ref_depth, 0)
            + "#" * self.is_unborrowed_ref
            + " (borrowed)" * self.is_borrowed
        )

        # Return everything (that's available).
        # TODO this should return something like "T&&, where typedef T = ...".
        if full and self._ref_depth == self._typedef_alias_ref_depth:
            name = f"typedef {self._typedef_alias} = " if self._typedef_alias else ""
            name += self.definition.user_facing_name
            name += suffix

            return name

        # If this is the product of a typedef, return the name given.
        if self._typedef_alias:
            return prefix + self._typedef_alias + suffix

        # Fall back to the type definition.
        return prefix + self.definition.user_facing_name + suffix

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Type)

        return (
            self.definition == other.definition
            and self.is_unborrowed_ref == other.is_unborrowed_ref
            and self.ref_depth == other.ref_depth
        )

    @property
    def alignment(self) -> int:
        # FIXME replace magic number.
        return 8 if self.is_pointer else self.definition.alignment

    @property
    def size(self) -> int:
        # FIXME replace magic number.
        return 8 if self.is_pointer else self.definition.size

    @property
    def ir_definition(self) -> str:
        # Opaque pointer type.
        return "ptr" if self.is_pointer else self.definition.ir_definition

    @property
    def ir_type(self) -> str:
        # Opaque pointer type.
        if self.is_pointer:
            return "ptr"

        if self._typedef_alias and self._ref_depth == self._typedef_alias_ref_depth:
            return self.definition.get_ir_type(self.mangled_name)

        return self.definition.get_ir_type(None)

    def get_ir_initial_type_def(self) -> list[str]:
        # Not a typedef, nothing to do.
        if not self._typedef_alias:
            return []

        named_ref = self.definition.get_ir_type(self.mangled_name)
        return [f"{named_ref} = type {self.ir_definition}"]

    def to_value_type(self) -> "Type":
        value_type = self.copy()
        value_type._ref_depth = 0
        value_type.is_unborrowed_ref = False
        value_type.is_borrowed = False

        return value_type

    @property
    def mangled_name(self) -> str:
        assert not self.is_unborrowed_ref

        alias = self._typedef_alias or self.definition.mangled_name
        value_type_mangled = self.mangle_generic_type(alias, self._specialization)

        return (
            f"__RT{self.ref_depth}{value_type_mangled}__{self.ref_depth}TR"
            if self.is_reference
            else value_type_mangled
        )

    def to_ir_constant(self, value: str) -> str:
        # We shouldn't be able to initialize a pointer type with a constant.
        assert not self.is_pointer

        return self.definition.to_ir_constant(value)

    def copy(self) -> "Type":
        # FIXME should this be a deepcopy()?
        return copy(self)

    def to_reference(self) -> "Type":
        assert not self.is_unborrowed_ref

        ref_type = self.copy()
        ref_type._ref_depth += 1

        return ref_type

    def to_dereferenced_type(self) -> "Type":
        assert not self.is_unborrowed_ref
        assert self.is_reference

        deref_type = self.copy()
        deref_type._ref_depth -= 1

        return deref_type

    def to_unborrowed_ref(self) -> "Type":
        assert not self.is_unborrowed_ref

        unborrowed_type = self.copy()
        unborrowed_type.is_unborrowed_ref = True

        return unborrowed_type

    def to_borrowed_ref(self) -> "Type":
        # is_unborrowed_ref decays into a reference when borrowed.
        assert self.is_unborrowed_ref
        assert not self.is_borrowed

        borrowed_type = self.copy()
        borrowed_type._ref_depth += 1
        borrowed_type.is_borrowed = True
        borrowed_type.is_unborrowed_ref = False

        return borrowed_type

    def to_decayed_type(self) -> "Type":
        # Decay into a reference without borrowing.
        assert self.is_unborrowed_ref
        assert not self.is_borrowed

        decayed_type = self.copy()
        decayed_type._ref_depth += 1
        decayed_type.is_unborrowed_ref = False

        return decayed_type

    def without_borrowing(self) -> "Type":
        without_borrowing_type = self.copy()
        without_borrowing_type.is_borrowed = False
        without_borrowing_type.is_unborrowed_ref = False

        return without_borrowing_type

    def new_from_typedef(
        self, typedef_alias: str, specialization: list["Type"]
    ) -> "Type":
        new_type = self.copy()
        new_type._set_typedef_alias(typedef_alias)
        new_type._specialization = specialization

        return new_type


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

        self._name = name
        self.type = var_type
        self.constant = constant

        self.ir_reg: Optional[int] = None

    @cached_property
    def user_facing_name(self) -> str:
        return self._name

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name}: {repr(self.type)})"


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
    def __init__(self, expr_type: Type) -> None:
        super().__init__()

        self.type = expr_type
        self.result_reg: Optional[int] = None

    def is_return_guaranteed(self) -> bool:
        # At the moment no TypedExpressions can return
        return False

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

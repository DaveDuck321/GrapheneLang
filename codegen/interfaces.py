from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Iterator, Optional


@dataclass
class CompileTimeConstant:
    value: int

    def __str__(self) -> str:
        return str(self.value)


class TypeDefinition(ABC):
    @abstractmethod
    def graphene_literal_to_ir_constant(self, value: str) -> str:
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
        specialization: Optional[list["Type"]] = None,
    ) -> None:
        self.definition = definition
        self._specialization = specialization if specialization is not None else []

        self._typedef_alias = typedef_alias
        self._is_borrowed_reference = False

    @property
    def is_borrowed_reference(self) -> bool:
        return self._is_borrowed_reference

    @property
    def is_void(self) -> bool:
        return self.definition.is_void

    @property
    def generic_annotation(self) -> str:
        if not self._specialization:
            return ""

        generic_names = [arg.get_user_facing_name(True) for arg in self._specialization]
        return f"<{', '.join(generic_names)}>"

    def __repr__(self) -> str:
        name = f"{self._typedef_alias} = " if self._typedef_alias else ""
        name += repr(self.definition)

        return (
            f"{self.__class__.__name__}({name}, is_ref={self._is_borrowed_reference})"
        )

    def get_specialization(self) -> list["Type"]:
        return self._specialization.copy()

    def get_user_facing_name(self, full: bool) -> str:
        suffix = self.generic_annotation + ("&" * self._is_borrowed_reference)

        # Return everything (that's available).
        # TODO this should return something like "T&&, where typedef T = ...".
        if full:
            name = f"typedef {self._typedef_alias} = " if self._typedef_alias else ""
            name += self.definition.user_facing_name
            name += suffix
            return name

        # If this is the product of a typedef, return the name given.
        if self._typedef_alias:
            return self._typedef_alias + suffix

        # Fall back to the type definition.
        return self.definition.user_facing_name + suffix

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Type)

        return (
            self.definition == other.definition
            and self.is_borrowed_reference == other.is_borrowed_reference
        )

    @property
    def alignment(self) -> int:
        # FIXME replace magic number.
        return 8 if self.is_borrowed_reference else self.definition.alignment

    @property
    def size(self) -> int:
        # FIXME replace magic number.
        return 8 if self.is_borrowed_reference else self.definition.size

    @property
    def ir_definition(self) -> str:
        # Opaque pointer type.
        return "ptr" if self.is_borrowed_reference else self.definition.ir_definition

    @property
    def ir_type(self) -> str:
        # Opaque pointer type.
        if self.is_borrowed_reference:
            return "ptr"

        if self._typedef_alias:
            return self.definition.get_ir_type(self.mangled_name)

        return self.definition.get_ir_type(None)

    def get_ir_initial_type_def(self) -> list[str]:
        # Not a typedef, nothing to do.
        if not self._typedef_alias:
            return []

        named_ref = self.definition.get_ir_type(self.mangled_name)
        return [f"{named_ref} = type {self.ir_definition}"]

    @property
    def mangled_name(self) -> str:
        alias = self._typedef_alias or self.definition.mangled_name
        value_type_mangled = self.mangle_generic_type(alias, self._specialization)

        return (
            f"__RT{value_type_mangled}__TR"
            if self.is_borrowed_reference
            else value_type_mangled
        )

    def graphene_literal_to_ir_constant(self, value: str) -> str:
        # We shouldn't be able to initialize a reference type with a constant.
        assert not self.is_borrowed_reference
        return self.definition.graphene_literal_to_ir_constant(value)

    def copy(self) -> "Type":
        # FIXME should this be a deepcopy()?
        return copy(self)

    def take_reference(self) -> "Type":
        assert not self.is_borrowed_reference

        new_type = self.copy()
        new_type._is_borrowed_reference = True
        return new_type

    def convert_to_value_type(self) -> "Type":
        new_type = self.copy()
        new_type._is_borrowed_reference = False
        return new_type

    def new_from_typedef(
        self, typedef_alias: str, specialization: list["Type | CompileTimeConstant"]
    ) -> "Type":
        new_type = self.copy()
        new_type._typedef_alias = typedef_alias
        new_type._specialization = specialization

        return new_type

SpecializationItem = Type | CompileTimeConstant
GenericMapping = dict[str, SpecializationItem]

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
    def __init__(self, name: Optional[str], var_type: Type, constant: bool) -> None:
        super().__init__()

        self._name = name
        self.type = var_type
        self.constant = constant

        self.ir_reg: Optional[int] = None

    @property
    def user_facing_name(self) -> str:
        assert self._name is not None
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
        expr_type: Type,
        is_indirect_pointer_to_type: bool,
        was_reference_type_at_any_point: bool = False,
    ) -> None:
        super().__init__()
        # It is the callers responsibility to escape double indirections
        if expr_type.is_borrowed_reference:
            assert not is_indirect_pointer_to_type

        self.underlying_type = expr_type
        self.is_indirect_pointer_to_type = is_indirect_pointer_to_type

        # Used for better error messages
        self.was_reference_type_at_any_point = was_reference_type_at_any_point

        self.result_reg: Optional[int] = None

    def get_equivalent_pure_type(self) -> Type:
        if self.is_indirect_pointer_to_type:
            return self.underlying_type.take_reference()
        return self.underlying_type

    @property
    def has_address(self) -> bool:
        return (
            self.underlying_type.is_borrowed_reference
            or self.is_indirect_pointer_to_type
        )

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

    def get_user_facing_name(self, full: bool) -> str:
        return self.underlying_type.get_user_facing_name(full)

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

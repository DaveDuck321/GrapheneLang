from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable

from .interfaces import Type, SpecializationItem
from .builtin_types import (
    AnonymousType,
    HeapArrayDefinition,
    NamedType,
    StackArrayDefinition,
    StructDefinition,
)


@dataclass
class GenericArgument:
    name: str
    is_value_arg: bool


@dataclass
class UnresolvedType:
    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    @abstractmethod
    def get_typedef_dependencies(self) -> list[str]:
        pass

    @abstractmethod
    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> "UnresolvedType":
        pass

    @abstractmethod
    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        pass

    @abstractmethod
    def pattern_match(
        self, target: Type, mapping_out: "UnresolvedGenericMapping"
    ) -> bool:
        pass


@dataclass
class CompileTimeConstant:
    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    @abstractmethod
    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> "CompileTimeConstant":
        pass

    @abstractmethod
    def resolve(self) -> int:
        pass

    @abstractmethod
    def pattern_match(
        self, target: int, mapping_out: "UnresolvedGenericMapping"
    ) -> bool:
        pass


UnresolvedSpecializationItem = UnresolvedType | CompileTimeConstant
UnresolvedGenericMapping = dict[str, UnresolvedSpecializationItem]


@dataclass
class NumericLiteralConstant(CompileTimeConstant):
    value: int

    def format_for_output_to_user(self) -> str:
        return str(self.value)

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> CompileTimeConstant:
        return NumericLiteralConstant(self.value)

    def resolve(self) -> int:
        return self.value

    def pattern_match(self, target: int, mapping_out: UnresolvedGenericMapping) -> bool:
        return target == self.value


@dataclass
class GenericValueReference(CompileTimeConstant):
    argument_name: str

    def format_for_output_to_user(self) -> str:
        return f"@{self.argument_name}"

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> CompileTimeConstant:
        # TODO: user facing errors
        assert self.argument_name in specialization_map

        specialized_value = specialization_map[self.argument_name]
        assert isinstance(specialized_value, int)
        return NumericLiteralConstant(specialized_value)

    def resolve(self) -> int:
        assert False

    def pattern_match(self, target: int, mapping_out: UnresolvedGenericMapping) -> bool:
        if self.argument_name in mapping_out:
            # TODO: user facing error
            return mapping_out[self.argument_name] == target

        mapping_out[self.argument_name] = NumericLiteralConstant(target)
        return True


@dataclass
class UnresolvedTypeWrapper(UnresolvedType):
    resolved_type: Type

    def format_for_output_to_user(self) -> str:
        return self.resolved_type.format_for_output_to_user()

    def get_typedef_dependencies(self) -> list[str]:
        return []

    def produce_specialized_copy(
        self, _: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        return UnresolvedTypeWrapper(self.resolved_type)

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        return self.resolved_type

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        return self.resolved_type == target


@dataclass
class UnresolvedNamedType(UnresolvedType):
    name: str
    specialization: list[UnresolvedSpecializationItem]

    def format_for_output_to_user(self) -> str:
        if len(self.specialization) == 0:
            return self.name

        specialization_format = ", ".join(
            arg.format_for_output_to_user() for arg in self.specialization
        )
        return f"{self.name}<{specialization_format}>"

    def get_typedef_dependencies(self) -> list[str]:
        specialization_depends = []
        for specialization in self.specialization:
            if isinstance(specialization, UnresolvedType):
                specialization_depends.extend(specialization.get_typedef_dependencies())

        return [self.name, *specialization_depends]

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        result = UnresolvedNamedType(self.name, [])
        for item in self.specialization:
            result.specialization.append(
                item.produce_specialized_copy(specialization_map)
            )

        return result

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_specialization = []
        for specialization in self.specialization:
            if isinstance(specialization, UnresolvedType):
                resolved_specialization.append(specialization.resolve(lookup))
            else:
                assert isinstance(specialization, CompileTimeConstant)
                resolved_specialization.append(specialization.resolve())

        return lookup(self.name, resolved_specialization)

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        if not isinstance(target, NamedType):
            return False

        if len(self.specialization) != len(target.specialization):
            return False

        for this_arg, target_arg in zip(self.specialization, target.specialization):
            if isinstance(this_arg, CompileTimeConstant) != isinstance(target_arg, int):
                return False

            if not this_arg.pattern_match(target_arg, mapping_out):
                return False

        return True


@dataclass
class UnresolvedGenericType(UnresolvedType):
    name: str

    def format_for_output_to_user(self) -> str:
        return self.name

    def get_typedef_dependencies(self) -> list[str]:
        return [self.name]

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        # TODO: user facing errors
        assert self.name in specialization_map

        specialized_type = specialization_map[self.name]
        assert isinstance(specialized_type, Type)
        return UnresolvedTypeWrapper(specialized_type)

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        assert False

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        if self.name in mapping_out:
            # TODO: user facing error
            return target == mapping_out[self.name]

        mapping_out[self.name] = UnresolvedTypeWrapper(target)
        return True


@dataclass
class UnresolvedReferenceType(UnresolvedType):
    value_type: UnresolvedType

    def format_for_output_to_user(self) -> str:
        return self.value_type.format_for_output_to_user() + "&"

    def get_typedef_dependencies(self) -> list[UnresolvedType]:
        return []

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        return UnresolvedReferenceType(
            self.value_type.produce_specialized_copy(specialization_map)
        )

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        # TODO: support circular references
        return self.value_type.resolve(lookup).convert_to_reference_type()

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        if not target.is_reference:
            return False

        return self.value_type.pattern_match(
            target.convert_to_value_type(), mapping_out
        )


@dataclass
class UnresolvedStructType(UnresolvedType):
    members: list[tuple[str, UnresolvedType]]

    def format_for_output_to_user(self) -> str:
        member_format = ", ".join(
            f"{member_name}: {member_type.format_for_output_to_user()}"
            for member_name, member_type in self.members
        )
        return "{" + ", ".join(member_format) + "}"

    def get_typedef_dependencies(self) -> list[str]:
        dependencies = []
        for _, member_type in self.members:
            if isinstance(member_type, UnresolvedType):
                dependencies.extend(member_type.get_typedef_dependencies())
        return dependencies

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        result = UnresolvedStructType([])
        for name, member_type in self.members:
            result.members.append(
                (
                    name,
                    member_type.produce_specialized_copy(specialization_map),
                )
            )
        return result

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_members = []
        for member_name, member in self.members:
            resolved_members.append((member_name, member.resolve(lookup)))

        return AnonymousType(StructDefinition(resolved_members))

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        assert False  # TODO


@dataclass
class UnresolvedStackArrayType(UnresolvedType):
    member_type: UnresolvedType
    dimensions: list[CompileTimeConstant]

    def format_for_output_to_user(self) -> str:
        member_format = self.member_type.format_for_output_to_user()
        dimensions_format = ", ".join(
            dimension.format_for_output_to_user() for dimension in self.dimensions
        )
        return f"{member_format}[{dimensions_format}]"

    def get_typedef_dependencies(self) -> list[str]:
        return self.member_type.get_typedef_dependencies()

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        result = UnresolvedStackArrayType(
            self.member_type.produce_specialized_copy(specialization_map), []
        )
        for dimension in self.dimensions:
            result.dimensions.append(
                dimension.produce_specialized_copy(specialization_map)
            )
        return result

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_dimensions = []
        for dimension in self.dimensions:
            resolved_dimensions.append(dimension.resolve())

        resolved_member = self.member_type.resolve(lookup)
        return AnonymousType(StackArrayDefinition(resolved_member, resolved_dimensions))

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        assert False  # TODO


@dataclass
class UnresolvedHeapArrayType(UnresolvedType):
    member_type: UnresolvedType
    known_dimensions: list[CompileTimeConstant]

    def format_for_output_to_user(self) -> str:
        member_format = self.member_type.format_for_output_to_user()
        if len(self.known_dimensions) == 0:
            return f"{member_format}[&]"

        dimensions_format = ", ".join(
            dimension.format_for_output_to_user() for dimension in self.known_dimensions
        )
        return f"{member_format}[&, {dimensions_format}]"

    def get_typedef_dependencies(self) -> list[str]:
        return self.member_type.get_typedef_dependencies()

    def produce_specialized_copy(
        self, specialization_map: dict[str, SpecializationItem]
    ) -> UnresolvedType:
        result = UnresolvedHeapArrayType(
            self.member_type.produce_specialized_copy(specialization_map), []
        )
        for dimension in self.known_dimensions:
            result.known_dimensions.append(
                dimension.produce_specialized_copy(specialization_map)
            )
        return result

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_dimensions = []
        for dimension in self.known_dimensions:
            resolved_dimensions.append(dimension.resolve())

        resolved_member = self.member_type.resolve(lookup)
        return AnonymousType(HeapArrayDefinition(resolved_member, resolved_dimensions))

    def pattern_match(
        self, target: Type, mapping_out: dict[str, UnresolvedSpecializationItem]
    ) -> bool:
        assert False  # TODO


@dataclass
class SpecializedTypedef:
    name: str
    specialization: list[UnresolvedSpecializationItem]
    aliased: UnresolvedType


@dataclass
class GenericTypedef:
    name: str
    generics: list[GenericArgument]
    aliased: UnresolvedType


class TypeSymbolTable:
    def __init__(self) -> None:
        self._generic_unresolved_types: dict[str, GenericTypedef] = {}
        self._unresolved_types: dict[str, list[SpecializedTypedef]] = defaultdict(list)
        self._unresolved_visiting_state: dict[str, int] = defaultdict(int)

        self._resolved_types: dict[str, list[NamedType]] = defaultdict(list)

    def get_dependencies_for_typedef(self, typedef_name: str) -> list[str]:
        dependencies: list[str] = []
        if typedef_name in self._unresolved_types:
            for node in self._unresolved_types[typedef_name]:
                dependencies.extend(node.aliased.get_typedef_dependencies())

                for specialization in node.specialization:
                    if not isinstance(specialization, UnresolvedType):
                        continue

                    dependencies.extend(specialization.get_typedef_dependencies())

        if typedef_name in self._generic_unresolved_types:
            generic_definition = self._generic_unresolved_types[typedef_name].aliased
            dependencies.extend(generic_definition.get_typedef_dependencies())

        return dependencies

    def sort_dependencies_topologically(self, name: str, order: list[str]) -> None:
        if self._unresolved_visiting_state[name] == 2:
            return  # Already finished

        # TODO: user facing error message (cyclic type)
        assert self._unresolved_visiting_state[name] == 0
        self._unresolved_visiting_state[name] = 1  # In-progress

        # Recurse into dependencies
        typedef_dependencies = self.get_dependencies_for_typedef(name)
        for dependency in typedef_dependencies:
            self.sort_dependencies_topologically(dependency, order)

        self._unresolved_visiting_state[name] = 2  # Finished
        order.append(name)

    def specialize_and_resolve_generic_type(
        self, name: str, specialization: list[SpecializationItem]
    ) -> NamedType:
        generic = self._generic_unresolved_types[name]

        specialization_map = {}
        # Check specialization arguments are the correct type
        for generic_arg, specialization_item in zip(generic.generics, specialization):
            if isinstance(specialization_item, int):
                # TODO: proper user-facing error
                assert generic_arg.is_value_arg
            else:
                assert isinstance(specialization_item, Type)
                # TODO: proper user-facing error
                assert not generic_arg.is_value_arg

            specialization_map[generic_arg.name] = specialization_item

        unresolved_alias = generic.aliased.produce_specialized_copy(specialization_map)
        resolved_alias = self.resolve_type(unresolved_alias)
        result = NamedType(
            generic.name, specialization, resolved_alias.definition, resolved_alias
        )
        self._resolved_types[generic.name].append(result)
        return result

    def resolve_all_types(self) -> None:
        # Guarantee all typedefed specializations have been resolved
        resolution_order = []
        for unresolved_type_name in self._unresolved_types:
            self.sort_dependencies_topologically(unresolved_type_name, resolution_order)

        for type_name in resolution_order:
            resolved_specialization: list[SpecializationItem] = []
            # TODO: further sort this list to parse a single type in the correct order
            for unresolved_type in self._unresolved_types[type_name]:
                for arg in unresolved_type.specialization:
                    if isinstance(arg, CompileTimeConstant):
                        resolved_specialization.append(arg.resolve())
                    else:
                        assert isinstance(arg, UnresolvedType)
                        resolved_specialization.append(self.resolve_type(arg))

                resolved_rhs = self.resolve_type(unresolved_type.aliased)
                self._resolved_types[type_name].append(
                    NamedType(
                        type_name,
                        resolved_specialization,
                        resolved_rhs.definition,
                        resolved_rhs,
                    )
                )

        # Ensure we only resolve each type once
        self._unresolved_types.clear()

    def lookup_type(
        self, prefix: str, specialization: list[SpecializationItem]
    ) -> NamedType:
        candidates = []
        for resolved_type in self._resolved_types[prefix]:
            if resolved_type.specialization == specialization:
                candidates.append(resolved_type)

        # TODO: user facing error
        assert len(candidates) <= 1

        if len(candidates) != 0:
            return candidates[0]

        # We haven't found a specialization so we produce one from the the generic
        # TODO: user facing error
        assert prefix in self._generic_unresolved_types
        return self.specialize_and_resolve_generic_type(prefix, specialization)

    def add_generic_unresolved_type(self, unresolved_typedef: GenericTypedef) -> None:
        # TODO: user facing error
        assert unresolved_typedef.name not in self._generic_unresolved_types
        self._generic_unresolved_types[unresolved_typedef.name] = unresolved_typedef

    def add_unresolved_type(self, unresolved_typedef: SpecializedTypedef) -> None:
        # TODO: check for duplicates
        self._unresolved_types[unresolved_typedef.name].append(unresolved_typedef)

    def add_resolved_type(self, resolved_type: NamedType) -> None:
        # TODO: check for duplicates
        self._resolved_types[resolved_type.name].append(resolved_type)

    def resolve_type(self, unresolved_type: UnresolvedType) -> Type:
        return unresolved_type.resolve(partial(TypeSymbolTable.lookup_type, self))

    def get_ir_for_initialization(self) -> list[str]:
        ir: list[str] = []
        for type_name in self._resolved_types:
            for defined_type in self._resolved_types[type_name]:
                if defined_type.alias is None:
                    continue  # Skip over primitive types

                ir.append(
                    f"{defined_type.ir_type} = type {defined_type.definition.ir_type}"
                )
        return ir

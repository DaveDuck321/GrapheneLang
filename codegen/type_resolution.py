from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from typing import Callable, Iterable, Optional

from .builtin_types import (
    AnonymousType,
    FunctionSignature,
    HeapArrayDefinition,
    NamedType,
    PrimitiveType,
    StackArrayDefinition,
    StructDefinition,
)
from .expressions import InitializerList
from .interfaces import (
    GenericArgument,
    GenericMapping,
    SpecializationItem,
    Type,
    TypedExpression,
    do_specializations_match,
    format_arguments,
    format_generics,
    format_specialization,
)
from .type_conversions import get_implicit_conversion_cost
from .user_facing_errors import (
    AmbiguousFunctionCall,
    BuiltinSourceLocation,
    DoubleReferenceError,
    ErrorWithLocationInfo,
    FailedLookupError,
    GrapheneError,
    Location,
    MultipleTypeDefinitions,
    NonDeterminableSize,
    OverloadResolutionError,
    PatternMatchDeductionFailure,
    RedefinitionError,
    SpecializationFailed,
    SubstitutionFailure,
)


def get_cost_at_pattern_match_depth(depth: int) -> int:
    # TODO: maybe this should be a list for infinite precision?
    # ATM. only 16 generic deductions/ nested levels are allowed
    return 16 ** (16 - depth)


@dataclass(frozen=True)
class UnresolvedType:
    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    @abstractmethod
    def produce_specialized_copy(self, generics: GenericMapping) -> "UnresolvedType":
        pass

    @abstractmethod
    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        pass

    @abstractmethod
    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        pass

    @abstractmethod
    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        pass


@dataclass(frozen=True)
class CompileTimeConstant:
    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    @abstractmethod
    def produce_specialized_copy(
        self, generics: GenericMapping
    ) -> "CompileTimeConstant":
        pass

    @abstractmethod
    def resolve(self) -> int:
        pass

    @abstractmethod
    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        pass

    @abstractmethod
    def pattern_match(
        self, target: int, out: GenericMapping, depth: int
    ) -> Optional[int]:
        pass


UnresolvedSpecializationItem = UnresolvedType | CompileTimeConstant


@dataclass(frozen=True)
class NumericLiteralConstant(CompileTimeConstant):
    value: int

    def format_for_output_to_user(self) -> str:
        return str(self.value)

    def produce_specialized_copy(self, _: GenericMapping) -> CompileTimeConstant:
        return NumericLiteralConstant(self.value)

    def resolve(self) -> int:
        return self.value

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set()

    def pattern_match(
        self, target: int, _: GenericMapping, depth: int
    ) -> Optional[int]:
        if target != self.value:
            return None
        return 0


@dataclass(frozen=True)
class GenericValueReference(CompileTimeConstant):
    name: str

    @property
    def argument(self) -> GenericArgument:
        return GenericArgument(self.name, True)

    def format_for_output_to_user(self) -> str:
        return self.name

    def produce_specialized_copy(self, generics: GenericMapping) -> CompileTimeConstant:
        assert self.argument in generics.mapping

        specialized_value = generics.mapping[self.argument]
        if not isinstance(specialized_value, int):
            raise SpecializationFailed(
                self.format_for_output_to_user(),
                specialized_value.format_for_output_to_user(),
            )

        return NumericLiteralConstant(specialized_value)

    def resolve(self) -> int:
        assert False

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return {self.argument}

    def pattern_match(
        self, target: int, out: GenericMapping, depth: int
    ) -> Optional[int]:
        if self.argument in out.mapping:
            # TODO: user facing error
            if out.mapping[self.argument] != target:
                return None

        out.mapping[self.argument] = target
        return get_cost_at_pattern_match_depth(depth)


@dataclass(frozen=True)
class UnresolvedTypeWrapper(UnresolvedType):
    resolved_type: Type

    def format_for_output_to_user(self) -> str:
        return self.resolved_type.format_for_output_to_user()

    def produce_specialized_copy(self, _: GenericMapping) -> UnresolvedType:
        return UnresolvedTypeWrapper(self.resolved_type)

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        return self.resolved_type

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set()

    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        if self.resolved_type != target:
            return None
        return 0


@dataclass(frozen=True)
class UnresolvedNamedType(UnresolvedType):
    name: str
    specialization: tuple[UnresolvedSpecializationItem, ...]

    def format_for_output_to_user(self) -> str:
        if len(self.specialization) == 0:
            return self.name

        specialization_format = ", ".join(
            arg.format_for_output_to_user() for arg in self.specialization
        )
        return f"{self.name}<{specialization_format}>"

    def produce_specialized_copy(self, generics: GenericMapping) -> UnresolvedType:
        specialization = []
        for item in self.specialization:
            specialization.append(item.produce_specialized_copy(generics))

        return UnresolvedNamedType(self.name, tuple(specialization))

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_specialization = []
        for specialization in self.specialization:
            if isinstance(specialization, UnresolvedType):
                resolved_specialization.append(specialization.resolve(lookup))
            else:
                assert isinstance(specialization, CompileTimeConstant)
                resolved_specialization.append(specialization.resolve())

        return lookup(self.name, resolved_specialization)

    def _pattern_match_impl(
        self, target: Type, mapping_out: GenericMapping, depth: int
    ) -> Optional[int]:
        if not isinstance(target, NamedType):
            return None

        # If our name doesn't match, recurse into the target's alias
        if target.name != self.name:
            if target.alias is None:
                return None

            return self.pattern_match(target.alias, mapping_out, depth)

        if len(self.specialization) != len(target.specialization):
            return None

        cost = 0
        for this_arg, target_arg in zip(self.specialization, target.specialization):
            if isinstance(this_arg, CompileTimeConstant) != isinstance(target_arg, int):
                return None

            result: Optional[int] = None
            if isinstance(this_arg, CompileTimeConstant):
                assert isinstance(target_arg, int)
                result = this_arg.pattern_match(target_arg, mapping_out, depth + 1)

            if isinstance(this_arg, UnresolvedType):
                assert isinstance(target_arg, Type)
                result = this_arg.pattern_match(target_arg, mapping_out, depth + 1)

            if result is None:
                return None

            cost += result

        return cost

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set().union(
            *(
                item.get_generics_taking_part_in_pattern_match()
                for item in self.specialization
            )
        )

    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        curr = target

        while curr:
            curr_mapping = GenericMapping(out.mapping.copy(), out.pack.copy())
            result = self._pattern_match_impl(curr, curr_mapping, depth)

            if result is not None:
                out.mapping.update(curr_mapping.mapping)
                out.pack = curr_mapping.pack
                return result

            curr = curr.alias if isinstance(curr, NamedType) else None

        return None


@dataclass(frozen=True)
class UnresolvedGenericType(UnresolvedType):
    name: str

    @property
    def argument(self) -> GenericArgument:
        return GenericArgument(self.name, False)

    def format_for_output_to_user(self) -> str:
        return self.name

    def produce_specialized_copy(self, generics: GenericMapping) -> UnresolvedType:
        assert self.argument in generics.mapping

        specialized_type = generics.mapping[self.argument]
        if not isinstance(specialized_type, Type):
            raise SpecializationFailed(
                self.format_for_output_to_user(), str(specialized_type)
            )

        return UnresolvedTypeWrapper(specialized_type)

    def resolve(self, _: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        assert False

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return {self.argument}

    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        if self.name in out.mapping:
            if target != out.mapping[self.argument]:
                return None

        out.mapping[self.argument] = target
        return get_cost_at_pattern_match_depth(depth)


@dataclass(frozen=True)
class UnresolvedReferenceType(UnresolvedType):
    value_type: UnresolvedType

    def format_for_output_to_user(self) -> str:
        return self.value_type.format_for_output_to_user() + "&"

    def produce_specialized_copy(self, generics: GenericMapping) -> UnresolvedType:
        return UnresolvedReferenceType(
            self.value_type.produce_specialized_copy(generics)
        )

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        # TODO: support circular references
        resolved_value = self.value_type.resolve(lookup)
        if resolved_value.is_reference:
            raise DoubleReferenceError(resolved_value.format_for_output_to_user(True))

        return resolved_value.convert_to_reference_type()

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return self.value_type.get_generics_taking_part_in_pattern_match()

    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        if not target.is_reference:
            return None

        return self.value_type.pattern_match(
            target.convert_to_value_type(), out, depth + 1
        )


@dataclass(frozen=True)
class UnresolvedStructType(UnresolvedType):
    members: tuple[tuple[str, UnresolvedType], ...]

    def format_for_output_to_user(self) -> str:
        member_format = ", ".join(
            f"{member_name}: {member_type.format_for_output_to_user()}"
            for member_name, member_type in self.members
        )
        return "{" + ", ".join(member_format) + "}"

    def produce_specialized_copy(self, generics: GenericMapping) -> UnresolvedType:
        return UnresolvedStructType(
            tuple(
                (name, member_type.produce_specialized_copy(generics))
                for name, member_type in self.members
            )
        )

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_members = []
        for member_name, member in self.members:
            resolved_members.append((member_name, member.resolve(lookup)))

        return AnonymousType(StructDefinition(resolved_members))

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set().union(
            *(
                member[1].get_generics_taking_part_in_pattern_match()
                for member in self.members
            )
        )

    def pattern_match(self, _1: Type, _2: GenericMapping, _3: int) -> Optional[int]:
        assert False  # TODO


@dataclass(frozen=True)
class UnresolvedStackArrayType(UnresolvedType):
    member_type: UnresolvedType
    dimensions: tuple[CompileTimeConstant, ...]

    def format_for_output_to_user(self) -> str:
        member_format = self.member_type.format_for_output_to_user()
        dimensions_format = ", ".join(
            dimension.format_for_output_to_user() for dimension in self.dimensions
        )
        return f"{member_format}[{dimensions_format}]"

    def produce_specialized_copy(self, generics: GenericMapping) -> UnresolvedType:
        return UnresolvedStackArrayType(
            self.member_type.produce_specialized_copy(generics),
            tuple(dim.produce_specialized_copy(generics) for dim in self.dimensions),
        )

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_dimensions = []
        for dimension in self.dimensions:
            resolved_dimensions.append(dimension.resolve())

        resolved_member = self.member_type.resolve(lookup)
        return AnonymousType(StackArrayDefinition(resolved_member, resolved_dimensions))

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set().union(
            self.member_type.get_generics_taking_part_in_pattern_match(),
            *(
                dim.get_generics_taking_part_in_pattern_match()
                for dim in self.dimensions
            ),
        )

    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        if not isinstance(target.definition, StackArrayDefinition):
            return None

        if len(self.dimensions) != len(target.definition.dimensions):
            return None

        cost = 0
        for target_dim, our_dim in zip(target.definition.dimensions, self.dimensions):
            result = our_dim.pattern_match(target_dim, out, depth + 1)
            if result is None:
                return None
            cost += result

        result = self.member_type.pattern_match(
            target.definition.member, out, depth + 1
        )
        if result is None:
            return None

        return cost + result


@dataclass(frozen=True)
class UnresolvedHeapArrayType(UnresolvedType):
    member_type: UnresolvedType
    known_dimensions: tuple[CompileTimeConstant, ...]

    def format_for_output_to_user(self) -> str:
        member_format = self.member_type.format_for_output_to_user()
        if len(self.known_dimensions) == 0:
            return f"{member_format}[&]"

        dimensions_format = ", ".join(
            dimension.format_for_output_to_user() for dimension in self.known_dimensions
        )
        return f"{member_format}[&, {dimensions_format}]"

    def produce_specialized_copy(self, generics: GenericMapping) -> UnresolvedType:
        return UnresolvedHeapArrayType(
            self.member_type.produce_specialized_copy(generics),
            tuple(
                dim.produce_specialized_copy(generics) for dim in self.known_dimensions
            ),
        )

    def resolve(self, lookup: Callable[[str, list[SpecializationItem]], Type]) -> Type:
        resolved_dimensions = []
        for dimension in self.known_dimensions:
            resolved_dimensions.append(dimension.resolve())

        resolved_member = self.member_type.resolve(lookup)
        return AnonymousType(HeapArrayDefinition(resolved_member, resolved_dimensions))

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set().union(
            self.member_type.get_generics_taking_part_in_pattern_match(),
            *(
                dim.get_generics_taking_part_in_pattern_match()
                for dim in self.known_dimensions
            ),
        )

    def pattern_match(
        self, target: Type, out: GenericMapping, depth: int
    ) -> Optional[int]:
        if not isinstance(target.definition, HeapArrayDefinition):
            return None

        if len(self.known_dimensions) != len(target.definition.known_dimensions):
            return None

        cost = 0
        for target_dim, our_dim in zip(
            target.definition.known_dimensions, self.known_dimensions
        ):
            result = our_dim.pattern_match(target_dim, out, depth + 1)
            if result is None:
                return None

        result = self.member_type.pattern_match(
            target.definition.member, out, depth + 1
        )
        if result is None:
            return None
        return cost + result


@dataclass(frozen=True)
class Typedef:
    name: str
    generics: tuple[GenericArgument, ...]
    expanded_specialization: tuple[UnresolvedSpecializationItem, ...]
    aliased: UnresolvedType
    loc: Location

    @staticmethod
    def construct(
        name: str,
        generics: tuple[GenericArgument, ...],
        specialization: Iterable[UnresolvedSpecializationItem],
        aliased: UnresolvedType,
        loc: Location,
    ):
        explicitly_matched_generics = set().union(
            *(
                item.get_generics_taking_part_in_pattern_match()
                for item in specialization
            )
        )

        unmatched_generics = [
            generic
            for generic in generics
            if generic not in explicitly_matched_generics
        ]

        expanded_specialization = [*specialization]
        # Atm we don't support both generics and pattern matching simultaneously
        # TODO: remove this
        if len(generics) != 0:
            assert len(expanded_specialization) == 0

        for generic in unmatched_generics:
            if generic.is_value_arg:
                expanded_specialization.append(GenericValueReference(generic.name))
            else:
                expanded_specialization.append(UnresolvedGenericType(generic.name))

        return Typedef(name, generics, tuple(expanded_specialization), aliased, loc)

    def pattern_match(
        self,
        target_specialization: list[SpecializationItem],
        mapping_out: GenericMapping,
    ) -> Optional[int]:
        cost = 0
        for item, target in zip(
            self.expanded_specialization, target_specialization, strict=True
        ):
            if isinstance(item, CompileTimeConstant) != isinstance(target, int):
                return None

            if len(item.get_generics_taking_part_in_pattern_match()) == 0:
                # Nothing to match, we rely on type equality instead
                # This costs nothing
                continue

            result: Optional[int] = None
            if isinstance(item, CompileTimeConstant):
                assert isinstance(target, int)
                result = item.pattern_match(target, mapping_out, 1)

            if isinstance(item, UnresolvedType):
                assert isinstance(target, Type)
                result = item.pattern_match(target, mapping_out, 1)

            if result is None:
                return None

            cost += result

        return cost

    def produce_specialized_copy(self, generics: GenericMapping) -> "Typedef":
        new_specialization = tuple(
            item.produce_specialized_copy(generics)
            for item in self.expanded_specialization
        )
        new_alias = self.aliased.produce_specialized_copy(generics)

        return Typedef(self.name, tuple(), new_specialization, new_alias, self.loc)


@dataclass(frozen=True)
class UnresolvedFunctionSignature:
    name: str
    expanded_specialization: tuple[UnresolvedSpecializationItem, ...]
    arguments: tuple[UnresolvedType, ...]
    parameter_pack_argument_name: Optional[str]
    return_type: UnresolvedType

    def collapse_into_type(self, arg: TypedExpression | Type) -> Type:
        if isinstance(arg, InitializerList):
            raise NotImplementedError()
        if isinstance(arg, TypedExpression):
            return arg.underlying_type
        if isinstance(arg, Type):
            return arg

        assert False

    def format_for_output_to_user(self) -> str:
        specialization_str = ""
        if len(self.expanded_specialization) != 0:
            specialization_list = ", ".join(
                (
                    item.format_for_output_to_user()
                    for item in self.expanded_specialization
                ),
            )
            specialization_str = f"<{specialization_list}>"

        formatted_args = [arg.format_for_output_to_user() for arg in self.arguments]
        if self.parameter_pack_argument_name is not None:
            formatted_args.append(self.parameter_pack_argument_name + "...")

        args_str = ", ".join(formatted_args)
        return_str = self.return_type.format_for_output_to_user()
        return f"{self.name}{specialization_str} : ({args_str}) -> {return_str}"

    def get_generics_taking_part_in_pattern_match(self) -> set[GenericArgument]:
        return set().union(
            *(
                item.get_generics_taking_part_in_pattern_match()
                for item in self.expanded_specialization
            )
        )

    def produce_specialized_copy(
        self,
        generics: GenericMapping,
    ) -> "UnresolvedFunctionSignature":
        unmatched_generics = (
            self.get_generics_taking_part_in_pattern_match() - generics.mapping.keys()
        )
        if len(unmatched_generics) != 0:
            raise PatternMatchDeductionFailure(self.name, unmatched_generics.pop().name)

        if self.parameter_pack_argument_name is None:
            assert len(generics.pack) == 0

        new_specialization: list[UnresolvedSpecializationItem] = []
        for item in self.expanded_specialization:
            new_specialization.append(item.produce_specialized_copy(generics))

        arguments: list[UnresolvedType] = []
        for arg in self.arguments:
            arguments.append(arg.produce_specialized_copy(generics))

        unresolved_packed_types = [
            UnresolvedTypeWrapper(pack_type) for pack_type in generics.pack
        ]
        arguments.extend(unresolved_packed_types)
        new_specialization.extend(unresolved_packed_types)

        return UnresolvedFunctionSignature(
            self.name,
            tuple(new_specialization),
            tuple(arguments),
            None,  # We have just expanded the parameter pack
            self.return_type.produce_specialized_copy(generics),
        )

    def pattern_match(
        self,
        target_args: list[TypedExpression] | list[Type],
        target_specialization: list[SpecializationItem],
        out: GenericMapping,
        depth: int,
    ) -> Optional[int]:
        cost = 0

        # Match the specialization
        for target_item, item in zip(
            target_specialization, self.expanded_specialization
        ):
            if isinstance(item, CompileTimeConstant) != isinstance(target_item, int):
                return False

            if len(item.get_generics_taking_part_in_pattern_match()) == 0:
                continue  # Allow for implicit conversions + type equivalency

            result: Optional[int] = None
            if isinstance(item, CompileTimeConstant):
                assert isinstance(target_item, int)
                result = item.pattern_match(target_item, out, depth + 1)

            if isinstance(item, UnresolvedType):
                assert isinstance(target_item, Type)
                result = item.pattern_match(target_item, out, depth + 1)

            if result is None:
                return None

            cost += result

        # Check the argument count
        if self.parameter_pack_argument_name is None:
            if len(self.arguments) != len(target_args):
                return None
        else:
            if len(self.arguments) >= len(target_args):
                return None

        # Match the (non-packed) arguments
        for target_arg, unresolved_arg in zip(
            target_args[: len(self.arguments)], self.arguments
        ):
            if len(unresolved_arg.get_generics_taking_part_in_pattern_match()) == 0:
                continue  # Allow for implicit conversions + type equivalency

            if isinstance(target_arg, InitializerList):
                # See: `docs/types_overview.c3`
                # TODO: we should convert this to an anonymous struct
                #   for now we just ignore initializer lists
                continue

            arg_type = self.collapse_into_type(target_arg)
            result = unresolved_arg.pattern_match(arg_type, out, depth + 1)
            if result is None:
                return None

            cost += result

        # Match the packed arguments
        if self.parameter_pack_argument_name is not None:
            assert len(out.pack) == 0
            out.pack.extend(
                (
                    self.collapse_into_type(arg)
                    for arg in target_args[len(self.arguments) :]
                )
            )
            cost += get_cost_at_pattern_match_depth(depth + 1) * len(out.pack)

        return cost


@dataclass(frozen=True)
class FunctionDeclaration:
    is_foreign: bool
    arg_names: tuple[str, ...]
    pack_type_name: Optional[str]
    generics: tuple[GenericArgument, ...]
    signature: UnresolvedFunctionSignature
    loc: Location

    @staticmethod
    def construct(
        name: str,
        is_foreign: bool,
        generics: tuple[GenericArgument, ...],
        specialization: Iterable[UnresolvedSpecializationItem],
        arg_names: tuple[str, ...],
        pack_type_name: Optional[str],
        arg_types: tuple[UnresolvedType, ...],
        parameter_pack_argument_name: Optional[str],
        return_type: UnresolvedType,
        location: Location,
    ):
        explicitly_matched_generics = set().union(
            *(
                item.get_generics_taking_part_in_pattern_match()
                for item in specialization
            )
        )

        unmatched_generics = [
            generic
            for generic in generics
            if generic not in explicitly_matched_generics
        ]

        expanded_specialization = [*specialization]
        # Atm we don't support both generics and pattern matching simultaneously
        assert len(expanded_specialization) == 0  # TODO: remove this

        for generic in unmatched_generics:
            if generic.is_value_arg:
                expanded_specialization.append(GenericValueReference(generic.name))
            else:
                expanded_specialization.append(UnresolvedGenericType(generic.name))

        signature = UnresolvedFunctionSignature(
            name,
            tuple(expanded_specialization),
            arg_types,
            parameter_pack_argument_name,
            return_type,
        )
        return FunctionDeclaration(
            is_foreign, arg_names, pack_type_name, generics, signature, location
        )

    def format_for_output_to_user(self) -> str:
        prefix = "foreign" if self.is_foreign else "function"
        generic_format = format_generics(self.generics, self.pack_type_name)
        return f"{prefix}{generic_format} {self.signature.format_for_output_to_user()}"

    def pattern_match(
        self,
        target_args: list[TypedExpression] | list[Type],
        target_specialization: list[SpecializationItem],
        out: GenericMapping,
    ) -> Optional[int]:
        return self.signature.pattern_match(target_args, target_specialization, out, 0)

    def produce_specialized_copy(
        self, generics: GenericMapping
    ) -> UnresolvedFunctionSignature:
        return self.signature.produce_specialized_copy(generics)


class SymbolTable:
    """
    Definitions:
        `(un)resolved`: an unresolved type is produced directly by the parser,
        it may have placeholder generics and parameter packs. A resolved type
        has no generics and is used directly by codegen.

        `specialize`: before a type can be resolved, all its generic arguments
        must be expanded. Specializing an UnresolvedType performs this
        substitution: the result is still an UnresolvedType but it may now be
        resolved -- specialization does not fail with SFINAE since no lookup has
        been performed.

        `lookup(name, specialization)`: the symbol table considers all
        unresolved symbol definitions, it specializes, resolves, and returns the
        type/ function which best matches the provided specialization.
    """

    def __init__(self) -> None:
        self._unresolved_fndefs: dict[str, list[FunctionDeclaration]] = defaultdict(
            list
        )
        self._unresolved_typedefs: dict[str, list[Typedef]] = defaultdict(list)

        # Remember the order that symbols are added so we can give sane error messages
        self._newly_added_unresolved_typedefs: list[Typedef] = []
        self._newly_added_unresolved_functions: list[FunctionDeclaration] = []

        # Symbols requiring codegen + cache for future lookup
        self._resolved_types: dict[tuple[str, int], list[NamedType]] = defaultdict(list)
        self._resolved_functions: dict[str, list[FunctionSignature]] = defaultdict(list)
        self._remaining_to_codegen: list[
            tuple[FunctionDeclaration, FunctionSignature]
        ] = []

    def resolve_specialization_item(
        self, unresolved: UnresolvedSpecializationItem
    ) -> SpecializationItem:
        if isinstance(unresolved, UnresolvedType):
            return self.resolve_type(unresolved)
        return unresolved.resolve()

    def resolve_type(self, unresolved: UnresolvedType) -> Type:
        return unresolved.resolve(partial(SymbolTable.lookup_named_type, self))

    def resolve_named_type(
        self,
        name: str,
        unresolved_specialization: tuple[UnresolvedSpecializationItem, ...],
        alias: UnresolvedType,
    ) -> NamedType:
        specialization = [
            self.resolve_specialization_item(item) for item in unresolved_specialization
        ]

        # Is the type already in the cache?
        key = (name, len(unresolved_specialization))
        for other in self._resolved_types[key]:
            if do_specializations_match(other.specialization, specialization):
                return other

        # Else resolve it from scratch
        resolved_type = NamedType(name, specialization)
        try:
            # Prematurely assume type is resolvable (for cyclic types)
            self._resolved_types[key].append(resolved_type)
            resolved_type.update_with_finalized_alias(self.resolve_type(alias))
            if not resolved_type.is_finite:
                raise NonDeterminableSize(resolved_type.format_for_output_to_user(True))
        except SubstitutionFailure as exc:
            # Type is NOT resolvable, undo bad assumption and rethrow error
            self._resolved_types[key].remove(resolved_type)
            raise exc

        return resolved_type

    def resolve_function(
        self, fn: UnresolvedFunctionSignature, declaration: FunctionDeclaration
    ) -> FunctionSignature:
        resolved_specialization: list[SpecializationItem] = []
        for arg in fn.expanded_specialization:
            if isinstance(arg, CompileTimeConstant):
                resolved_specialization.append(arg.resolve())
            else:
                assert isinstance(arg, UnresolvedType)
                resolved_specialization.append(self.resolve_type(arg))

        resolved_arguments: list[Type] = []
        for arg_type in fn.arguments:
            resolved_arguments.append(self.resolve_type(arg_type))

        resolved_return = self.resolve_type(fn.return_type)

        # Check the cache, have we already resolved this function?
        for other in self._resolved_functions[fn.name]:
            if (
                do_specializations_match(other.specialization, resolved_specialization)
                and other.arguments == resolved_arguments
                and other.return_type == resolved_return
            ):
                return other

        result = FunctionSignature(
            fn.name,
            resolved_arguments,
            resolved_return,
            resolved_specialization,
            declaration.is_foreign,
        )
        self._resolved_functions[fn.name].append(result)
        self._remaining_to_codegen.append((declaration, result))
        return result

    def select_function_with_least_implicit_conversion_cost(
        self,
        fn_name: str,
        candidate_functions: list[tuple[FunctionSignature, FunctionDeclaration]],
        fn_args: list[TypedExpression] | list[Type],
    ) -> Optional[FunctionSignature]:
        functions_by_cost: list[tuple[int, FunctionSignature, FunctionDeclaration]] = []

        for function, declaration in candidate_functions:
            total_cost = 0
            if len(fn_args) != len(function.arguments):
                continue

            for src_expr, dest_type in zip(fn_args, function.arguments):
                cost = get_implicit_conversion_cost(src_expr, dest_type)
                if cost is not None:
                    total_cost += cost
                else:
                    break  # Conversion failed!
            else:
                # Discard this candidate if conversion fails
                functions_by_cost.append((total_cost, function, declaration))

        functions_by_cost.sort(key=lambda t: t[0])
        best_functions = [
            fn[1:] for fn in functions_by_cost if fn[0] == functions_by_cost[0][0]
        ]

        if len(best_functions) == 0:
            return None

        if len(best_functions) == 1:
            return best_functions[0][0]

        # If there are two or more equally good candidates, then this function
        # call is ambiguous.
        raise AmbiguousFunctionCall(
            f"{fn_name}{format_arguments(fn_args)}",
            [(fn.format_for_output_to_user(), fn.loc) for _, fn in best_functions],
        )

    def lookup_function(
        self,
        name: str,
        given_specialization: list[SpecializationItem],
        args: list[TypedExpression] | list[Type],
    ) -> FunctionSignature:
        # Specializes and resolves the function corresponding to the correct definition
        # There are two types of function definition:
        #   1) function [T] fn : ...            (pure generic)
        #   2a) function [T] fn<Thing<T>> : ... (pattern match a)
        #   2b) function [T] fn : (a : T) ...   (pattern match b)
        #
        # The first (pure generic) is syntax sugar for the following:
        #     function [T] fn  ==> function [T] fn<T>
        #
        # A function may deduce its specialization based on its arguments
        #   For now, either the full specialization must be given or none at all
        #   TODO: relax this requirement
        #
        # We choose which function gets priority based on the following criteria:
        #   1) When initialized the signature MUST not create a substitution failure
        #   2) All parameter implicit conversions MUST succeed
        #   3) Minimize the number of top level generic deductions
        #   4) Minimize the number of 2nd level generic deductions
        #   N) Minimize the number of N level generic deductions
        #   N+1) Minimize the cost of implicit conversion

        if name not in self._unresolved_fndefs:
            # Substitution impossible
            raise FailedLookupError(
                "function", f"function {name}{format_arguments(args)}"
            )

        all_candidates: list[
            tuple[int, UnresolvedFunctionSignature, FunctionDeclaration]
        ] = []
        for candidate in self._unresolved_fndefs[name]:
            generics = GenericMapping({}, [])
            cost = candidate.pattern_match(args, given_specialization, generics)
            if cost is None:
                continue

            # We've matched the pattern, now we specialize the candidate
            # TODO: catch these errors
            specialized = candidate.produce_specialized_copy(generics)
            all_candidates.append((cost, specialized, candidate))

        # Find the cheapest valid candidate (doing overload resolution if cost is the same)
        all_candidates.sort(key=lambda item: item[0])
        for _, candidates in groupby(all_candidates, key=lambda item: item[0]):
            resolved_candidates: list[
                tuple[FunctionSignature, FunctionDeclaration]
            ] = []
            for _, candidate, declaration in candidates:
                try:
                    resolved_candidates.append(
                        (self.resolve_function(candidate, declaration), declaration)
                    )
                except SubstitutionFailure:
                    pass  # SFINAE

            if len(resolved_candidates) != 0:
                # TODO: pass along the specialization for error messages
                signature = self.select_function_with_least_implicit_conversion_cost(
                    name, resolved_candidates, args
                )
                if signature is not None:
                    return signature

        raise OverloadResolutionError(
            f"{name}{format_specialization(given_specialization)}{format_arguments(args)}",
            [fn.format_for_output_to_user() for fn in self._unresolved_fndefs[name]],
        )

    def lookup_named_type(
        self, prefix: str, specialization: list[SpecializationItem]
    ) -> Type:
        # Specializes and resolves the type corresponding to the correct typedef
        # There are two types of typedef:
        #   1) typedef [T] MyType : ...            (pure generic)
        #   2) typedef [T] MyType<Thing<T>> : ...  (pattern match)
        #
        # The first (pure generic) is syntax sugar for the following:
        #     typedef [U, V] MyType  ==> typedef [U, V] MyType<U, V>
        #
        # We choose which typedef gets priority based on the following criteria:
        #   1) The specialization argument count MUST match
        #   2) When initialized the type MUST not create a substitution failure
        #   3) Minimize the number of top level generic deductions
        #   4) Minimize the number of 2nd level generic deductions
        #   N) Minimize the number of N level generic deductions

        if prefix not in self._unresolved_typedefs:
            # Substitution impossible
            raise FailedLookupError(
                "type",
                f"typedef {prefix}{format_specialization(specialization)} : ...",
            )

        all_candidates: list[tuple[int, Typedef]] = []
        for candidate in self._unresolved_typedefs[prefix]:
            if len(candidate.expanded_specialization) != len(specialization):
                continue

            # Deduce the generic mapping by comparing (un)resolved specializations
            generics = GenericMapping({}, [])
            cost = candidate.pattern_match(specialization, generics)

            if cost is None:
                continue

            # We've matched the pattern, now we specialize the candidate
            # TODO: catch these errors
            specialized = candidate.produce_specialized_copy(generics)
            all_candidates.append((cost, specialized))

        # Find the cheapest valid candidate (doing overload resolution if cost is the same)
        all_candidates.sort(key=lambda item: item[0])
        for _, candidates in groupby(all_candidates, key=lambda item: item[0]):
            resolved_candidates: list[tuple[Type, Location]] = []
            for _, candidate in candidates:
                try:
                    resolved_type = self.resolve_named_type(
                        candidate.name,
                        candidate.expanded_specialization,
                        candidate.aliased,
                    )

                    if do_specializations_match(
                        resolved_type.specialization, specialization
                    ):
                        resolved_candidates.append((resolved_type, candidate.loc))
                except SubstitutionFailure:
                    pass  # SFINAE
                except GrapheneError as e:
                    raise ErrorWithLocationInfo(
                        e.message, candidate.loc, "typedef"
                    ) from e

            if len(resolved_candidates) == 0:
                continue
            if len(resolved_candidates) == 1:
                return resolved_candidates[0][0]

            specialization_str = format_specialization(specialization)
            raise MultipleTypeDefinitions(
                f"{prefix}{specialization_str}",
                [
                    (candidate.format_for_output_to_user(True), loc)
                    for candidate, loc in resolved_candidates
                ],
            )

        raise SubstitutionFailure(prefix + format_specialization(specialization))

    def add_function(self, fn_to_add: FunctionDeclaration) -> None:
        matched_functions = self._unresolved_fndefs[fn_to_add.signature.name]

        for target in matched_functions:
            if target.is_foreign or fn_to_add.is_foreign:
                raise RedefinitionError(
                    "non-overloadable foreign function",
                    target.signature.format_for_output_to_user(),
                )

        matched_functions.append(fn_to_add)
        self._newly_added_unresolved_functions.append(fn_to_add)

    def add_builtin_type(self, type_to_add: PrimitiveType) -> None:
        self._resolved_types[type_to_add.name, 0].append(type_to_add)
        self.add_type(
            Typedef(
                type_to_add.name,
                tuple(),
                tuple(),
                UnresolvedNamedType(type_to_add.name, tuple()),
                BuiltinSourceLocation(),
            )
        )

    def add_type(self, type_to_add: Typedef) -> None:
        # TODO: check for duplicates etc.
        self._unresolved_typedefs[type_to_add.name].append(type_to_add)
        self._newly_added_unresolved_typedefs.append(type_to_add)

    def resolve_all_non_generics(self) -> None:
        for typedef in self._newly_added_unresolved_typedefs:
            if len(typedef.generics) != 0:
                continue  # We can't lookup this type because it isn't fully specialized yet

            try:
                resolved_specialization = [
                    self.resolve_specialization_item(item)
                    for item in typedef.expanded_specialization
                ]
                self.lookup_named_type(typedef.name, resolved_specialization)
            except GrapheneError as e:
                raise ErrorWithLocationInfo(e.message, typedef.loc, "typedef") from e

        for fn in self._newly_added_unresolved_functions:
            if len(fn.generics) != 0 or fn.pack_type_name is not None:
                continue

            try:
                resolved_specialization = [
                    self.resolve_specialization_item(item)
                    for item in fn.signature.expanded_specialization
                ]
                resolved_args = [
                    self.resolve_type(item) for item in fn.signature.arguments
                ]
                self.lookup_function(
                    fn.signature.name, resolved_specialization, resolved_args
                )
            except GrapheneError as e:
                raise ErrorWithLocationInfo(
                    e.message, fn.loc, "function declaration"
                ) from e

        self._newly_added_unresolved_typedefs.clear()
        self._newly_added_unresolved_functions.clear()

    def get_next_function_to_codegen(
        self,
    ) -> Optional[tuple[FunctionDeclaration, FunctionSignature]]:
        try:
            return self._remaining_to_codegen.pop()
        except IndexError:
            return None

    def get_ir_for_initialization(self) -> list[str]:
        ir: list[str] = []
        for type_name in self._resolved_types:
            for defined_type in self._resolved_types[type_name]:
                if defined_type.alias is None:
                    continue  # Skip over primitive types

                if defined_type.should_defer_to_alias_for_ir():
                    continue

                ir.append(
                    f"{defined_type.ir_type} = type {defined_type.definition.ir_type}"
                )

        for fn_name in self._resolved_functions:
            for signature in self._resolved_functions[fn_name]:
                if signature.is_foreign:
                    ir.append(signature.generate_declaration_ir())

        return ir

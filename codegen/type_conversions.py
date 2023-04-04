from abc import abstractmethod
from functools import cached_property
from typing import Iterator, Optional

from .builtin_types import (
    ArrayDefinition,
    StructDefinition,
    IntegerDefinition,
    UnresolvedType,
)
from .interfaces import Type, TypedExpression
from .user_facing_errors import (
    CannotAssignToInitializerList,
    InvalidInitializerListConversion,
    InvalidInitializerListLength,
    OperandError,
    TypeCheckerError,
)


class StructInitializer(TypedExpression):
    def __init__(self, struct_type: Type, member_exprs: list[TypedExpression]) -> None:
        assert not struct_type.is_borrowed_reference
        assert isinstance(struct_type.definition, StructDefinition)
        assert len(member_exprs) == len(struct_type.definition.members)

        self._result_ref: Optional[str] = None
        self._members: list[TypedExpression] = []
        self._conversion_exprs: list[TypedExpression] = []

        for target_member_type, member_expr in zip(
            struct_type.definition.members, member_exprs, strict=True
        ):
            member, extra_exprs = do_implicit_conversion(
                member_expr, target_member_type.type
            )
            self._members.append(member)
            self._conversion_exprs.extend(extra_exprs)

        super().__init__(struct_type, False, False)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        ir: list[str] = self.expand_ir(self._conversion_exprs, reg_gen)

        previous_ref = "undef"
        for index, value in enumerate(self._members):
            current_ref = f"%{next(reg_gen)}"
            ir.append(
                f"{current_ref} = insertvalue {self.underlying_type.ir_type} {previous_ref}, "
                f"{value.ir_ref_with_type_annotation}, {index}"
            )

            previous_ref = current_ref

        self.result_ref = previous_ref
        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self.result_ref

    def __repr__(self) -> str:
        return f"StructInitializer({self.underlying_type}, {self._members})"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot modify temporary struct")


class InitializerList(TypedExpression):
    @abstractmethod
    def get_user_facing_name(self, full: bool) -> str:
        pass

    def get_equivalent_pure_type(self) -> Type:
        assert False

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert False

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise CannotAssignToInitializerList()

    @abstractmethod
    def convert_to_type(self, other: Type) -> tuple[int, list[TypedExpression]]:
        pass


class NamedInitializerList(InitializerList):
    def __init__(self, members: list[TypedExpression], names: list[str]) -> None:
        super().__init__(UnresolvedType(), False, False)

        self._members = dict(zip(names, members, strict=True))

    def get_user_facing_name(self, full: bool) -> str:
        members = [
            f"{name}: {type_name.underlying_type.get_user_facing_name(full)}"
            for name, type_name in self._members.items()
        ]
        return f"{{{', '.join(members)}}}"

    def __repr__(self) -> str:
        return f"InitializerList({list(self._members.items())})"

    def convert_to_type(self, other: Type) -> tuple[int, list[TypedExpression]]:
        error_message = InvalidInitializerListConversion(
            other.get_user_facing_name(False), self.get_user_facing_name(False)
        )

        if not isinstance(other.definition, StructDefinition):
            raise error_message

        if len(other.definition.members) != len(self._members):
            raise InvalidInitializerListLength(
                len(self._members), len(other.definition.members)
            )

        ordered_members: list[TypedExpression] = []
        for member in other.definition.members:
            if member.name not in self._members:
                raise error_message

            ordered_members.append(self._members[member.name])

        # TODO: remember cost during struct conversions
        return 0, [StructInitializer(other, ordered_members)]


class UnnamedInitializerList(InitializerList):
    def __init__(self, members: list[TypedExpression]) -> None:
        super().__init__(UnresolvedType(), False, False)

        self._members = members

    def get_user_facing_name(self, full: bool) -> str:
        type_names = [
            member.underlying_type.get_user_facing_name(full)
            for member in self._members
        ]
        return f"{{{', '.join(type_names)}}}"

    def __repr__(self) -> str:
        return f"InitializerList({self._members})"

    def convert_to_type(self, other: Type) -> tuple[int, list[TypedExpression]]:
        error_message = InvalidInitializerListConversion(
            other.get_user_facing_name(False), self.get_user_facing_name(False)
        )

        if not isinstance(other.definition, StructDefinition):
            raise error_message

        if len(other.definition.members) != len(self._members):
            raise InvalidInitializerListLength(
                len(self._members), len(other.definition.members)
            )

        # TODO: remember cost during struct conversions
        return 0, [StructInitializer(other, self._members)]


class SquashIntoUnderlyingType(TypedExpression):
    def __init__(self, ref: TypedExpression) -> None:
        # Converts a TypedExpression into an underlying type with no indirection
        super().__init__(ref.underlying_type, False)

        self.ref = ref

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#load-instruction

        self.result_reg = next(reg_gen)
        return_type_ir = self.ref.underlying_type.ir_type

        # <result> = load [volatile] <ty>, ptr <pointer>[, align <alignment>]...
        return [
            f"%{self.result_reg} = load {return_type_ir}, "
            f"{self.ref.ir_ref_with_type_annotation}, "
            f"align {self.get_equivalent_pure_type().alignment}"
        ]

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"SquashIntoUnderlyingType({self.ref})"

    def assert_can_read_from(self) -> None:
        self.ref.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot modify a squashed value")


class PromoteInteger(TypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
        src_definition = src.underlying_type.definition

        assert not src.has_address
        assert not dest_type.is_borrowed_reference
        assert isinstance(src_definition, IntegerDefinition)
        assert isinstance(dest_type.definition, IntegerDefinition)
        assert src_definition.is_signed == dest_type.definition.is_signed
        assert src_definition.bits < dest_type.definition.bits

        super().__init__(dest_type, False)

        self.src = src
        self.is_signed = src_definition.is_signed

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#sext-to-instruction
        # https://llvm.org/docs/LangRef.html#zext-to-instruction

        self.result_reg = next(reg_gen)

        instruction = "sext" if self.is_signed else "zext"

        # <result> = {s,z}ext <ty> <value> to <ty2> ; yields ty2
        return [
            f"%{self.result_reg} = {instruction} "
            f"{self.src.ir_ref_with_type_annotation} to {self.underlying_type.ir_type}"
        ]

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"PromoteInteger({self.src.underlying_type} to {self.underlying_type})"

    def assert_can_read_from(self) -> None:
        self.src.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        # TODO this isn't very helpful.
        raise OperandError("cannot modify promoted integers")


class Reinterpret(TypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
        # Bit cast between anything
        super().__init__(dest_type, False)

        self._src = src
        self._no_conversion_needed = (
            self._src.get_equivalent_pure_type().ir_type
            == self.get_equivalent_pure_type().ir_type
        )

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#bitcast-to-instruction

        if self._no_conversion_needed:
            return []

        self.result_reg = next(reg_gen)
        return [
            f"%{self.result_reg} = bitcast {self._src.ir_ref_with_type_annotation} "
            f"to {self.get_equivalent_pure_type().ir_type}"
        ]

    @property
    def ir_ref_without_type_annotation(self) -> str:
        if self._no_conversion_needed:
            return self._src.ir_ref_without_type_annotation
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"Reinterpret({self._src.underlying_type} to {self.underlying_type})"

    def assert_can_read_from(self) -> None:
        self._src.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._src.assert_can_write_to()


def implicit_conversion_impl(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[int, list[TypedExpression]]:
    """Attempt to convert expression src to type dest_type.

    Only the following conversions are allowed:
    - dereference an (non-reference) variable with an address to a value
    - initializer list -> compatible struct
    - integer promotion.
    - float promotion (TODO).

    Multiple conversions may be performed (e.g. dereference and then promote).

    If conversion is not possible, then a user-facing exception is raised.

    Args:
        src (TypedExpression): expression to convert.
        dest_type (Type): desired type.
        context (str, optional): conversion context, used in error messages.
            Defaults to "".

    Returns:
        tuple[int, list[TypedExpression]]: Tuple contains the promotion cost,
            The list contains the chain of expressions required to convert the
            expression to dest_type. The first element of the list is always src.
    """
    expr_list = [src]

    def last_expr() -> TypedExpression:
        return expr_list[-1]

    def last_type() -> Type:
        return expr_list[-1].underlying_type

    promotion_cost: int = 0

    # Always dereference implicit addresses
    if src.is_indirect_pointer_to_type:
        expr_list.append(SquashIntoUnderlyingType(src))

    # The type-system reference should not be implicitly dereferenced
    if last_type().is_borrowed_reference != dest_type.is_borrowed_reference:
        maybe_missing_borrow = False
        if src.underlying_type == dest_type.convert_to_value_type():
            maybe_missing_borrow = src.was_reference_type_at_any_point

        raise TypeCheckerError(
            context,
            src.underlying_type.get_user_facing_name(False),
            dest_type.get_user_facing_name(False),
            maybe_missing_borrow,
        )

    # Initializer lists
    if isinstance(src, InitializerList):
        additional_cost, exprs = src.convert_to_type(dest_type)

        promotion_cost += additional_cost
        expr_list.extend(exprs)

    # Integer promotion.
    # TODO we might want to relax the is_signed == is_signed rule.
    last_def = last_type().definition
    dest_def = dest_type.definition
    if (
        isinstance(last_def, IntegerDefinition)
        and isinstance(dest_def, IntegerDefinition)
        and last_def.is_signed == dest_def.is_signed
        and last_def.bits < dest_def.bits
    ):
        promotion_cost += dest_def.bits // last_def.bits
        expr_list.append(PromoteInteger(last_expr(), dest_type))

    # Array reference equivalence
    if (
        isinstance(last_def, ArrayDefinition)
        and isinstance(dest_def, ArrayDefinition)
        and last_def.dimensions[1:] == dest_def.dimensions[1:]
        and last_def.dimensions[0] >= dest_def.dimensions[0]
    ):
        # TODO: promotion cost going from known size to smaller/ unknown size
        expr_list.append(Reinterpret(last_expr(), dest_type))

    # TODO float promotion.

    if last_type() != dest_type:
        raise TypeCheckerError(
            context,
            src.underlying_type.get_user_facing_name(False),
            dest_type.get_user_facing_name(False),
        )

    return promotion_cost, expr_list


def do_implicit_conversion(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[TypedExpression, list[TypedExpression]]:
    _, expr_list = implicit_conversion_impl(src, dest_type, context)

    return expr_list[-1], expr_list[1:]


def is_type_implicitly_convertible(src: TypedExpression, dest_type: Type) -> bool:
    try:
        implicit_conversion_impl(src, dest_type)
    except TypeCheckerError:
        return False

    return True


def assert_is_implicitly_convertible(
    expr: TypedExpression, target: Type, context: str
) -> None:
    # Just discard the return value. It will throw if the conversion fails.
    # TODO maybe we could cache the result for later.
    implicit_conversion_impl(expr, target, context)


def get_implicit_conversion_cost(
    src: TypedExpression, dest_type: Type
) -> Optional[int]:
    try:
        cost, _ = implicit_conversion_impl(src, dest_type)
        return cost
    except TypeCheckerError:
        return None

from functools import cached_property
from typing import Iterator, Optional

from .builtin_types import ArrayDefinition, IntegerDefinition
from .interfaces import Type, TypedExpression
from .user_facing_errors import OperandError, TypeCheckerError, assert_else_throw, throw


class Dereference(TypedExpression):
    def __init__(self, ref: TypedExpression) -> None:
        super().__init__(ref.type.to_dereferenced_type())

        self.ref = ref

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#load-instruction

        self.result_reg = next(reg_gen)

        # We have a pointer to a value. Now we need to load that value into a
        # register.
        # <result> = load [volatile] <ty>, ptr <pointer>[, align <alignment>]...
        return [
            f"%{self.result_reg} = load {self.type.ir_type}, "
            f"{self.ref.ir_ref_with_type_annotation}, align {self.type.alignment}"
        ]

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"Dereference({self.ref})"

    def assert_can_read_from(self) -> None:
        self.ref.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        throw(OperandError("Cannot modify a dereferenced value"))


class Decay(TypedExpression):
    def __init__(self, expr: TypedExpression) -> None:
        super().__init__(expr.type.to_decayed_type())

        self.expr = expr

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return self.expr.ir_ref_without_type_annotation

    def __repr__(self) -> str:
        return f"Decay({self.expr})"

    def assert_can_read_from(self) -> None:
        self.expr.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self.expr.assert_can_write_to()


class PromoteInteger(TypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
        assert not src.type.is_pointer
        assert not dest_type.is_pointer
        assert isinstance(src.type.definition, IntegerDefinition)
        assert isinstance(dest_type.definition, IntegerDefinition)
        assert src.type.definition.is_signed == dest_type.definition.is_signed
        assert src.type.definition.bits < dest_type.definition.bits

        super().__init__(dest_type)

        self.src = src
        self.is_signed = src.type.definition.is_signed

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#sext-to-instruction
        # https://llvm.org/docs/LangRef.html#zext-to-instruction

        self.result_reg = next(reg_gen)

        instruction = "sext" if self.is_signed else "zext"

        # <result> = {s,z}ext <ty> <value> to <ty2> ; yields ty2
        return [
            f"%{self.result_reg} = {instruction} "
            f"{self.src.ir_ref_with_type_annotation} to {self.type.ir_type}"
        ]

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"PromoteInteger({self.src.type} to {self.type})"

    def assert_can_read_from(self) -> None:
        self.src.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        # TODO this isn't very helpful.
        throw(OperandError("Cannot modify promoted integers"))


class Reinterpret(TypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
        super().__init__(dest_type)

        self._src = src
        self._no_conversion_needed = self._src.type.ir_type == self.type.ir_type

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#bitcast-to-instruction
        if self._no_conversion_needed:
            return []

        self.result_reg = next(reg_gen)
        return [
            f"%{self.result_reg} = bitcast {self._src.ir_ref_with_type_annotation} to {self.type.ir_type}"
        ]

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        if self._no_conversion_needed:
            return self._src.ir_ref_without_type_annotation
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"Reinterpret({self._src.type} to {self.type})"

    def assert_can_read_from(self) -> None:
        self._src.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._src.assert_can_write_to()


def dereference_to_value(src: TypedExpression) -> list[TypedExpression]:
    expr_list: list[TypedExpression] = [src]
    while expr_list[-1].type.is_reference:
        expr_list.append(Dereference(expr_list[-1]))

    return expr_list[1:]


def dereference_to_single_reference(src: TypedExpression) -> list[TypedExpression]:
    return dereference_to_value(src)[:-1]


def implicit_conversion_impl(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[tuple[int, int], list[TypedExpression]]:
    """Attempt to convert expression src to type dest_type.

    Only the following conversions are allowed:
    - decaying an unborrowed reference to a normal reference (without
      borrowing).
    - dereferencing any reference type.
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
        tuple[tuple[int, int], list[TypedExpression]]: Tuple contains the
            promotion cost, followed by the dereferencing cost. The list
            contains the chain of expressions required to convert the expression
            to dest_type. The first element of the list is always src.
    """
    expr_list = [src]

    def last_expr() -> TypedExpression:
        return expr_list[-1]

    def last_type() -> Type:
        return expr_list[-1].type

    promotion_cost: int = 0
    dereferencing_cost: int = 0

    # If src hasn't been borrowed, then we are forced to decay the unborrowed
    # reference into a normal reference, and hope that everything works out.
    if last_type().is_unborrowed_ref:
        expr_list.append(Decay(last_expr()))

    # If src hasn't been borrowed, then we can only read its value.
    ref_depth_required = dest_type.ref_depth if last_type().is_borrowed else 0

    # We are only allowed to dereference.
    assert_else_throw(
        last_type().ref_depth >= ref_depth_required,
        TypeCheckerError(
            context,
            src.type.get_user_facing_name(False),
            dest_type.get_user_facing_name(False),
        ),
    )

    dereferencing_cost = last_type().ref_depth - ref_depth_required
    for _ in range(dereferencing_cost):
        expr_list.append(Dereference(last_expr()))

    last_def = last_type().definition
    dest_def = dest_type.definition

    # Integer promotion.
    # TODO we might want to relax the is_signed == is_signed rule.
    if (
        not last_type().is_pointer
        and not dest_type.is_pointer
        and isinstance(last_def, IntegerDefinition)
        and isinstance(dest_def, IntegerDefinition)
        and last_def.is_signed == dest_def.is_signed
        and last_def.bits < dest_def.bits
    ):
        promotion_cost += dest_def.bits // last_def.bits
        expr_list.append(PromoteInteger(last_expr(), dest_type))

    # Array reference equivalence
    if (
        last_type().is_reference
        and dest_type.is_reference
        and isinstance(last_def, ArrayDefinition)
        and isinstance(dest_def, ArrayDefinition)
        and last_def.dimensions[1:] == dest_def.dimensions[1:]
        and last_def.dimensions[0] >= dest_def.dimensions[0]
    ):
        # TODO: promotion cost going from known size to smaller/ unknown size
        expr_list.append(Reinterpret(last_expr(), dest_type))

    # TODO float promotion.

    assert_else_throw(
        last_type() == dest_type,
        TypeCheckerError(
            context,
            src.type.get_user_facing_name(False),
            dest_type.get_user_facing_name(False),
        ),
    )

    return (promotion_cost, dereferencing_cost), expr_list


def do_implicit_conversion(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[TypedExpression, list[TypedExpression]]:
    _, expr_list = implicit_conversion_impl(src, dest_type, context)

    return expr_list[-1], expr_list[1:]


class Wrapper(TypedExpression):
    def __repr__(self) -> str:
        return f"Wrapper({repr(self.type)})"

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        assert False

    def assert_can_read_from(self) -> None:
        assert False

    def assert_can_write_to(self) -> None:
        assert False


def is_type_implicitly_convertible(src_type: Type, dest_type: Type) -> bool:
    try:
        implicit_conversion_impl(Wrapper(src_type), dest_type)
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
    src_type: Type, dest_type: Type
) -> Optional[tuple[int, int]]:
    try:
        costs, _ = implicit_conversion_impl(Wrapper(src_type), dest_type)
        return costs
    except TypeCheckerError:
        return None

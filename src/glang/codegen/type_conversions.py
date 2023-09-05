from functools import cached_property
from typing import Iterator, Optional

from .builtin_types import HeapArrayDefinition, IntegerDefinition, StackArrayDefinition
from .interfaces import Type, TypedExpression
from .user_facing_errors import OperandError, TypeCheckerError


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
        assert not dest_type.is_reference
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

    # Initializer lists (+ anything else that depends on the TypedExpression)
    additional_cost, exprs = src.try_convert_to_type(dest_type)
    promotion_cost += additional_cost
    expr_list.extend(exprs)

    # The type-system reference should not be implicitly dereferenced
    if last_type().is_reference != dest_type.is_reference:
        maybe_missing_borrow = False
        if src.underlying_type == dest_type.convert_to_value_type():
            maybe_missing_borrow = src.was_reference_type_at_any_point

        raise TypeCheckerError(
            context,
            src.underlying_type.format_for_output_to_user(),
            dest_type.format_for_output_to_user(),
            maybe_missing_borrow,
        )

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
        last_type().is_reference
        and dest_type.is_reference
        and isinstance(last_def, (HeapArrayDefinition, StackArrayDefinition))
        and isinstance(dest_def, (HeapArrayDefinition, StackArrayDefinition))
        and last_def.member == dest_def.member
    ):
        if (
            isinstance(last_def, HeapArrayDefinition)
            and isinstance(dest_def, HeapArrayDefinition)
            and last_def.known_dimensions == dest_def.known_dimensions
        ):
            expr_list.append(Reinterpret(last_expr(), dest_type))

        if (
            isinstance(last_def, StackArrayDefinition)
            and isinstance(dest_def, HeapArrayDefinition)
            and last_def.dimensions[1:] == dest_def.known_dimensions
        ):
            # TODO: promotion cost going from known size to smaller/ unknown size
            expr_list.append(Reinterpret(last_expr(), dest_type))

        if (
            isinstance(last_def, StackArrayDefinition)
            and isinstance(dest_def, StackArrayDefinition)
            and last_def.dimensions[1:] == dest_def.dimensions[1:]
            and last_def.dimensions[0] >= dest_def.dimensions[0]
        ):
            # TODO: promotion cost going from known size to smaller/ unknown size
            expr_list.append(Reinterpret(last_expr(), dest_type))

    # TODO float promotion.

    if last_type() != dest_type:
        raise TypeCheckerError(
            context,
            src.underlying_type.format_for_output_to_user(),
            dest_type.format_for_output_to_user(),
        )

    return promotion_cost, expr_list


def do_implicit_conversion(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[TypedExpression, list[TypedExpression]]:
    _, expr_list = implicit_conversion_impl(src, dest_type, context)

    return expr_list[-1], expr_list[1:]


def assert_is_implicitly_convertible(
    expr: TypedExpression, target: Type, context: str
) -> None:
    # Just discard the return value. It will throw if the conversion fails.
    # TODO maybe we could cache the result for later.
    implicit_conversion_impl(expr, target, context)


def get_implicit_conversion_cost(
    src: Type | TypedExpression, dest_type: Type
) -> Optional[int]:
    class Wrapper(TypedExpression):
        def __init__(self, expr_type: Type) -> None:
            super().__init__(expr_type, False, False)

        @property
        def ir_ref_without_type_annotation(self) -> str:
            assert False

        def assert_can_read_from(self) -> None:
            pass

        def assert_can_write_to(self) -> None:
            pass

        def __repr__(self) -> str:
            return self.underlying_type.format_for_output_to_user(False)

    try:
        expression = src if isinstance(src, TypedExpression) else Wrapper(src)
        cost, _ = implicit_conversion_impl(expression, dest_type)
        return cost
    except TypeCheckerError:
        return None

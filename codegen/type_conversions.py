from functools import cached_property
from typing import Iterator

from .builtin_types import IntegerDefinition
from .interfaces import Type, TypedExpression
from .user_facing_errors import OperandError, TypeCheckerError, assert_else_throw, throw


class Dereference(TypedExpression):
    def __init__(self, ref: TypedExpression) -> None:
        assert ref.type.is_reference

        super().__init__(ref.type.get_non_reference_type())

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
        self.ref.assert_can_write_to()


class PromoteInteger(TypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
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


def do_implicit_conversion(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[TypedExpression, list[TypedExpression]]:
    """Attempt to convert expr to type target.

    Only the following conversions are allowed:
    - from reference type to value type.
    - integer promotion.
    - float promotion (TODO).

    Multiple conversions may be performed (e.g. dereference and then promote).

    If conversion is not possible, then a user-facing exception is raised.

    Args:
        src (TypedExpression): expression to convert.
        dest_type (Type): desired type.

    Returns:
        tuple[TypedExpression, list[TypedExpression]]: The expression of the
            desired type, plus a list of expressions that need to be evaluated
            in order to perform the conversion.
    """
    expr_list: list[TypedExpression] = [src]

    # Same type, nothing to do.
    if src.type == dest_type:
        return expr_list[-1], expr_list[1:]

    # Check if we need to dereference the expression.
    if src.type.is_reference and not dest_type.is_reference:
        expr_list.append(Dereference(src))

    current_def = expr_list[-1].type.definition
    dest_def = dest_type.definition

    # Integer promotion.
    # TODO we might want to relax the is_signed == is_signed rule.
    if (
        isinstance(current_def, IntegerDefinition)
        and isinstance(dest_def, IntegerDefinition)
        and current_def.is_signed == dest_def.is_signed
        and current_def.bits < dest_def.bits
    ):
        expr_list.append(PromoteInteger(expr_list[-1], dest_type))

    # TODO float promotion.

    assert_else_throw(
        expr_list[-1].type == dest_type,
        TypeCheckerError(
            context,
            src.type.get_user_facing_name(False),
            dest_type.get_user_facing_name(False),
        ),
    )

    return expr_list[-1], expr_list[1:]


def is_type_implicitly_convertible(src_type: Type, dest_type: Type) -> bool:
    # TODO can we implement this using do_implicit_conversion()?
    if src_type == dest_type:
        return True

    if src_type.is_reference and not dest_type.is_reference:
        src_type = src_type.get_non_reference_type()

    current_def = src_type.definition
    dest_def = dest_type.definition

    # Integer promotion.
    if (
        isinstance(current_def, IntegerDefinition)
        and isinstance(dest_def, IntegerDefinition)
        and current_def.is_signed == dest_def.is_signed
        and current_def.bits < dest_def.bits
    ):
        return True

    # TODO float promotion.

    return src_type == dest_type


def assert_is_implicitly_convertible(
    expr: TypedExpression, target: Type, context: str
) -> None:
    # Just discard the return value. It will throw if the conversion fails.
    # TODO maybe we could cache the result for later.
    do_implicit_conversion(expr, target, context)

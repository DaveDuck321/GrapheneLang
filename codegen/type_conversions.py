from functools import cached_property
from typing import Iterator

from .builtin_types import ReferenceType
from .interfaces import Type, TypedExpression
from .user_facing_errors import TypeCheckerError, assert_else_throw


class Dereference(TypedExpression):
    def __init__(self, ref: TypedExpression) -> None:
        assert isinstance(ref.type.definition, ReferenceType.Definition)

        # TODO need a nicer interface.
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
            f"{self.ref.ir_ref}, align {self.type.align}"
        ]

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"Dereference({self.ref})"

    def assert_can_read_from(self) -> None:
        self.ref.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self.ref.assert_can_write_to()


def do_implicit_conversion(
    expr: TypedExpression, target: Type, context: str = ""
) -> tuple[TypedExpression, list[TypedExpression]]:
    """Attempt to convert expr to type target.

    Only the following conversions are allowed:
    - from refererence type to value type.
    - integer promotion (TODO).
    - float promotion (TODO).

    Multiple conversions may be performed (TODO, e.g. dereference and then
    promote).

    If conversion is not possible, then a user-facing exception is raised.

    Args:
        expr (TypedExpression): expression to convert.
        target (Type): desired type.

    Returns:
        tuple[TypedExpression, list[TypedExpression]]: The expression of the
            desired type, plus a list of expressions that need to be evaluated
            in order to perform the conversion.
    """
    expr_list: list[TypedExpression] = [expr]

    # Same type, nothing to do.
    if expr.type == target:
        return expr_list[-1], expr_list[1:]

    # Check if we need to dereference the expression.
    if expr.type.is_reference and not target.is_reference:
        expr_list.append(Dereference(expr))

    # TODO integer and float promotions.

    assert_else_throw(
        expr_list[-1].type == target,
        TypeCheckerError(context, expr.type.name, target.name),
    )

    return expr_list[-1], expr_list[1:]


def is_type_implicitly_convertible(from_type: Type, target_type: Type) -> bool:
    # TODO can we implement this using do_implicit_conversion()?
    if from_type == target_type:
        return True

    if from_type.is_reference and not target_type.is_reference:
        from_type = from_type.get_non_reference_type()

    # TODO integer and float promotions.

    return from_type == target_type


def assert_is_implicitly_convertible(
    expr: TypedExpression, target: Type, context: str
) -> None:
    # Just discard the return value. It will throw if the conversion fails.
    # TODO maybe we could cache the result for later.
    do_implicit_conversion(expr, target, context)

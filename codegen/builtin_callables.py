from functools import cached_property
from typing import Callable, Iterator

from .builtin_types import IntType
from .interfaces import TypedExpression
from .type_conversions import assert_is_implicitly_convertible, do_implicit_conversion
from .user_facing_errors import OperandError, throw


class AddExpression(TypedExpression):
    def __init__(self, arguments: list[TypedExpression]) -> None:
        lhs, rhs = arguments

        # FIXME handle multiple int types.
        super().__init__(IntType())
        assert_is_implicitly_convertible(lhs, IntType(), "builtin add")
        assert_is_implicitly_convertible(rhs, IntType(), "builtin add")

        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"Add({self._lhs} + {self._rhs})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction
        # FIXME handle multiple int types.
        conv_lhs, extra_exprs_lhs = do_implicit_conversion(self._lhs, IntType())
        conv_rhs, extra_exprs_rhs = do_implicit_conversion(self._rhs, IntType())

        ir_lines: list[str] = []
        ir_lines.extend(self.expand_ir(extra_exprs_lhs, reg_gen))
        ir_lines.extend(self.expand_ir(extra_exprs_rhs, reg_gen))

        self.result_reg = next(reg_gen)

        # <result> = add nuw nsw <ty> <op1>, <op2>  ; yields ty:result
        ir_lines.append(
            f"%{self.result_reg} = add nuw nsw {conv_lhs.ir_ref_with_type_annotation}, {conv_rhs.ir_ref_without_type_annotation}"
        )
        return ir_lines

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        throw(OperandError("Cannot assign to `__builtin_add(..., ...)`"))


def get_builtin_callables() -> dict[
    str, Callable[[list[TypedExpression]], TypedExpression]
]:
    return {
        "__builtin_add": AddExpression,
    }

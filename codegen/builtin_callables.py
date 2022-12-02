from functools import cached_property
from typing import Iterator, Callable

from .user_facing_errors import (
    OperandError,
    throw,
)

from .interfaces import TypedExpression


class AddExpression(TypedExpression):
    def __init__(self, arguments: list[TypedExpression]) -> None:
        lhs, rhs = arguments
        super().__init__(lhs.type)

        assert lhs.type == rhs.type
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"Add({self._lhs} + {self._rhs})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction
        self.result_reg = next(reg_gen)

        # <result> = add nuw nsw <ty> <op1>, <op2>  ; yields ty:result
        return [
            f"%{self.result_reg} = add nuw nsw {self._lhs.ir_ref}, {self._rhs.ir_ref_without_type}"
        ]

    @cached_property
    def ir_ref_without_type(self) -> str:
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

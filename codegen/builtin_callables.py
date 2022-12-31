from functools import cached_property
from typing import Callable, Iterator

from .builtin_types import GenericIntType, IntegerDefinition
from .interfaces import TypedExpression
from .type_conversions import do_implicit_conversion
from .user_facing_errors import OperandError


class AddExpression(TypedExpression):
    def __init__(self, arguments: list[TypedExpression]) -> None:
        lhs, rhs = arguments
        # This is not a user-facing function, we don't need sensible error messages
        assert isinstance(lhs.type, GenericIntType)
        assert isinstance(rhs.type, GenericIntType)
        assert lhs.type.definition == rhs.type.definition

        super().__init__(lhs.type.to_value_type())

        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"Add({self._lhs} + {self._rhs})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction
        conv_lhs, extra_exprs_lhs = do_implicit_conversion(self._lhs, self.type)
        conv_rhs, extra_exprs_rhs = do_implicit_conversion(self._rhs, self.type)

        ir_lines: list[str] = []
        ir_lines.extend(self.expand_ir(extra_exprs_lhs, reg_gen))
        ir_lines.extend(self.expand_ir(extra_exprs_rhs, reg_gen))

        self.result_reg = next(reg_gen)

        assert isinstance(self.type.definition, IntegerDefinition)
        overflow_ir = "nsw" if self.type.definition.is_signed else "nuw"

        # <result> = add [nuw] [nsw] <ty> <op1>, <op2>  ; yields ty:result
        ir_lines.append(
            f"%{self.result_reg} = add {overflow_ir} {conv_lhs.ir_ref_with_type_annotation},"
            f" {conv_rhs.ir_ref_without_type_annotation}"
        )
        return ir_lines

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_add(..., ...)`")


def get_builtin_callables() -> dict[
    str, Callable[[list[TypedExpression]], TypedExpression]
]:
    return {
        "__builtin_add": AddExpression,
    }

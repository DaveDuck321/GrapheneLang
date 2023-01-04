from functools import cached_property
from typing import Callable, Iterator

from .builtin_types import SizeType, GenericIntType, IntegerDefinition
from .expressions import ConstantExpression
from .interfaces import TypedExpression, Type
from .type_conversions import do_implicit_conversion
from .user_facing_errors import OperandError


class AddExpression(TypedExpression):
    def __init__(
        self, specialization: list[Type], arguments: list[TypedExpression]
    ) -> None:
        lhs, rhs = arguments
        # This is not a user-facing function, we don't need sensible error messages
        assert len(specialization) == 0
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


class AlignOfExpression(TypedExpression):
    def __init__(
        self, specialization: list[Type], arguments: list[TypedExpression]
    ) -> None:
        assert len(arguments) == 0
        assert len(specialization) == 1
        self._argument_type = specialization[0]
        self._result = ConstantExpression(
            self._argument_type, str(self._argument_type.alignment)
        )

        super().__init__(SizeType())

    def __repr__(self) -> str:
        return f"AlignOf({self._argument_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return self._result.generate_ir(reg_gen)

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return self._result.ir_ref_without_type_annotation

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_alignof<...>()`")


class NarrowExpression(TypedExpression):
    def __init__(
        self, specialization: list[Type], arguments: list[TypedExpression]
    ) -> None:
        (self._argument,) = arguments
        (return_type,) = specialization

        self._arg_value_type = self._argument.type.to_value_type()

        assert isinstance(self._arg_value_type, GenericIntType)
        assert isinstance(return_type, GenericIntType)
        assert isinstance(self._arg_value_type.definition, IntegerDefinition)
        assert isinstance(return_type.definition, IntegerDefinition)
        assert self._arg_value_type.definition.bits > return_type.definition.bits

        super().__init__(return_type)

    def __repr__(self) -> str:
        return f"Narrow({self._argument} to {self.type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        conv_arg, extra_exprs_arg = do_implicit_conversion(
            self._argument, self._arg_value_type
        )
        ir_lines: list[str] = self.expand_ir(extra_exprs_arg, reg_gen)

        self.result_reg = next(reg_gen)
        return [
            *ir_lines,
            f"%{self.result_reg} = trunc {conv_arg.ir_ref_with_type_annotation} to {self.type.ir_type}",
        ]

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_narrow<...>(...)`")


def get_builtin_callables() -> dict[
    str, Callable[[list[Type], list[TypedExpression]], TypedExpression]
]:
    return {
        "__builtin_add": AddExpression,
        "__builtin_alignof": AlignOfExpression,
        "__builtin_narrow": NarrowExpression,
    }

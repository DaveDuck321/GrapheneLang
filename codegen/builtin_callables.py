from functools import cached_property
from typing import Callable, Iterator, Type as PyType

from .builtin_types import SizeType, GenericIntType, IntegerDefinition
from .expressions import ConstantExpression
from .interfaces import TypedExpression, Type
from .type_conversions import do_implicit_conversion
from .user_facing_errors import OperandError


class ArithmeticExpression(TypedExpression):
    # TODO: floating point support
    SIGNED_IR = ""
    UNSIGNED_IR = ""
    USER_FACING_NAME = ""

    def __init__(
        self, specialization: list[Type], arguments: list[TypedExpression]
    ) -> None:
        lhs, rhs = arguments
        # This is not a user-facing function, we don't need sensible error messages
        assert self.SIGNED_IR is not None
        assert self.UNSIGNED_IR is not None
        assert len(specialization) == 0
        assert isinstance(lhs.type, GenericIntType)
        assert isinstance(rhs.type, GenericIntType)
        assert lhs.type.definition == rhs.type.definition

        super().__init__(lhs.type.to_value_type())

        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._lhs}, {self._rhs})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction (and below)
        conv_lhs, extra_exprs_lhs = do_implicit_conversion(self._lhs, self.type)
        conv_rhs, extra_exprs_rhs = do_implicit_conversion(self._rhs, self.type)

        ir_lines: list[str] = []
        ir_lines.extend(self.expand_ir(extra_exprs_lhs, reg_gen))
        ir_lines.extend(self.expand_ir(extra_exprs_rhs, reg_gen))

        self.result_reg = next(reg_gen)

        assert isinstance(self.type.definition, IntegerDefinition)
        ir = self.SIGNED_IR if self.type.definition.is_signed else self.UNSIGNED_IR

        # eg. for addition
        # <result> = add [nuw] [nsw] <ty> <op1>, <op2>  ; yields ty:result
        ir_lines.append(
            f"%{self.result_reg} = {ir} {conv_lhs.ir_ref_with_type_annotation}, "
            f"{conv_rhs.ir_ref_without_type_annotation}"
        )
        return ir_lines

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError(f"Cannot assign to `{self.USER_FACING_NAME}(..., ...)`")


class AddExpression(ArithmeticExpression):
    SIGNED_IR = "add nsw"
    UNSIGNED_IR = "add nuw"
    USER_FACING_NAME = "__builtin_add"


class SubExpression(ArithmeticExpression):
    SIGNED_IR = "sub nsw"
    UNSIGNED_IR = "sub nuw"
    USER_FACING_NAME = "__builtin_sub"


class MultiplyExpression(ArithmeticExpression):
    SIGNED_IR = "mul nsw"
    UNSIGNED_IR = "mul nuw"
    USER_FACING_NAME = "__builtin_multiply"


class DivideExpression(ArithmeticExpression):
    SIGNED_IR = "sdiv"
    UNSIGNED_IR = "udiv"
    USER_FACING_NAME = "__builtin_divide"


class RemainderExpression(ArithmeticExpression):
    SIGNED_IR = "srem"
    UNSIGNED_IR = "urem"
    USER_FACING_NAME = "__builtin_remainder"


class ShiftLeftExpression(ArithmeticExpression):
    SIGNED_IR = "shl"
    UNSIGNED_IR = "shl"
    USER_FACING_NAME = "__builtin_shift_left"


class ShiftRightExpression(ArithmeticExpression):
    # NOTE: this is non-obvious behavior and should be documented
    #       I've chosen to sign-extend signed bit-shifts
    SIGNED_IR = "ashr"
    UNSIGNED_IR = "lshr"
    USER_FACING_NAME = "__builtin_shift_right"


class BitwiseAndExpression(ArithmeticExpression):
    SIGNED_IR = "and"
    UNSIGNED_IR = "and"
    USER_FACING_NAME = "__builtin_bitwise_and"


class BitwiseOrExpression(ArithmeticExpression):
    SIGNED_IR = "or"
    UNSIGNED_IR = "or"
    USER_FACING_NAME = "__builtin_bitwise_or"


class BitwiseXorExpression(ArithmeticExpression):
    SIGNED_IR = "xor"
    UNSIGNED_IR = "xor"
    USER_FACING_NAME = "__builtin_bitwise_xor"


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

        to_definition = return_type.definition
        from_definition = self._arg_value_type.definition
        assert isinstance(to_definition, IntegerDefinition)
        assert isinstance(from_definition, IntegerDefinition)

        assert from_definition.bits > to_definition.bits
        assert from_definition.is_signed == to_definition.is_signed

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
    def get_arithmetic_builtin(expression_class: PyType[ArithmeticExpression]):
        return (expression_class.USER_FACING_NAME, expression_class)

    arithmetic_instructions = dict(
        map(
            get_arithmetic_builtin,
            [
                AddExpression,
                SubExpression,
                MultiplyExpression,
                DivideExpression,
                RemainderExpression,
                ShiftLeftExpression,
                ShiftRightExpression,
                BitwiseAndExpression,
                BitwiseOrExpression,
                BitwiseXorExpression,
            ],
        )
    )

    return {
        **arithmetic_instructions,
        "__builtin_alignof": AlignOfExpression,
        "__builtin_narrow": NarrowExpression,
    }

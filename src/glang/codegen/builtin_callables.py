from abc import abstractmethod
from typing import Iterator
from typing import Type as PyType

from .builtin_types import (
    BoolDefinition,
    BoolType,
    GenericIntType,
    IntegerDefinition,
    IPtrType,
    SizeType,
)
from .expressions import ConstantExpression
from .interfaces import SpecializationItem, Type, TypedExpression
from .type_conversions import do_implicit_conversion
from .user_facing_errors import OperandError


class BuiltinCallable(TypedExpression):
    @abstractmethod
    def __init__(
        self, specialization: list[SpecializationItem], arguments: list[TypedExpression]
    ) -> None:
        pass


class UnaryIntegerExpression(BuiltinCallable):
    IR_FORMAT_STR = ""
    USER_FACING_NAME = ""

    def __init__(
        self, specialization: list[SpecializationItem], arguments: list[TypedExpression]
    ) -> None:
        (self._arg,) = arguments

        assert self.IR_FORMAT_STR is not None
        assert len(specialization) == 0

        definition = self._arg.underlying_type.definition
        assert isinstance(definition, (IntegerDefinition, BoolDefinition))

        self._arg_type = self._arg.underlying_type.convert_to_value_type()
        TypedExpression.__init__(self, self._arg_type, False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._arg})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction (and below)
        conv, extra_exprs = do_implicit_conversion(self._arg, self._arg_type)

        ir_lines: list[str] = []
        ir_lines.extend(self.expand_ir(extra_exprs, reg_gen))

        self.result_reg = next(reg_gen)

        op = self.IR_FORMAT_STR.format_map(
            {
                "type": self._arg_type.ir_type,
                "arg": conv.ir_ref_without_type_annotation,
            }
        )

        ir_lines.append(f"%{self.result_reg} = {op}")
        return ir_lines

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError(f"cannot assign to `{self.USER_FACING_NAME}(..., ...)`")


class BitwiseNotExpression(UnaryIntegerExpression):
    USER_FACING_NAME = "__builtin_bitwise_not"
    IR_FORMAT_STR = "xor {type} -1, {arg}"


class UnaryMinusExpression(UnaryIntegerExpression):
    USER_FACING_NAME = "__builtin_minus"
    IR_FORMAT_STR = "sub {type} 0, {arg}"


class BasicIntegerExpression(BuiltinCallable):
    SIGNED_IR = ""
    UNSIGNED_IR = ""
    USER_FACING_NAME = ""

    @staticmethod
    @abstractmethod
    def get_result_type(arguments: list[TypedExpression]) -> Type:
        pass

    def __init__(
        self, specialization: list[SpecializationItem], arguments: list[TypedExpression]
    ) -> None:
        lhs, rhs = arguments
        # This is not a user-facing function, we don't need sensible error messages
        assert self.SIGNED_IR is not None
        assert self.UNSIGNED_IR is not None
        assert len(specialization) == 0
        lhs_definition = lhs.underlying_type.definition
        rhs_definition = rhs.underlying_type.definition
        assert isinstance(lhs_definition, (IntegerDefinition, BoolDefinition))
        assert isinstance(rhs_definition, (IntegerDefinition, BoolDefinition))
        assert lhs.underlying_type == rhs.underlying_type

        TypedExpression.__init__(self, self.get_result_type(arguments), False)

        self._arg_type = lhs.underlying_type.convert_to_value_type()
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._lhs}, {self._rhs})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction (and below)
        conv_lhs, extra_exprs_lhs = do_implicit_conversion(self._lhs, self._arg_type)
        conv_rhs, extra_exprs_rhs = do_implicit_conversion(self._rhs, self._arg_type)

        ir_lines: list[str] = []
        ir_lines.extend(self.expand_ir(extra_exprs_lhs, reg_gen))
        ir_lines.extend(self.expand_ir(extra_exprs_rhs, reg_gen))

        self.result_reg = next(reg_gen)

        if isinstance(self._arg_type.definition, IntegerDefinition):
            ir = (
                self.SIGNED_IR
                if self._arg_type.definition.is_signed
                else self.UNSIGNED_IR
            )
        else:
            assert isinstance(self._arg_type.definition, BoolDefinition)
            ir = self.UNSIGNED_IR

        # eg. for addition
        # <result> = add [nuw] [nsw] <ty> <op1>, <op2>  ; yields ty:result
        ir_lines.append(
            f"%{self.result_reg} = {ir} {conv_lhs.ir_ref_with_type_annotation}, "
            f"{conv_rhs.ir_ref_without_type_annotation}"
        )
        return ir_lines

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError(f"cannot assign to `{self.USER_FACING_NAME}(..., ...)`")


class ArithmeticExpression(BasicIntegerExpression):
    @staticmethod
    def get_result_type(arguments: list[TypedExpression]) -> Type:
        return arguments[0].underlying_type.convert_to_value_type()


class AddExpression(ArithmeticExpression):
    SIGNED_IR = "add nsw"
    UNSIGNED_IR = "add nuw"
    USER_FACING_NAME = "__builtin_add"


class SubExpression(ArithmeticExpression):
    SIGNED_IR = "sub nsw"
    UNSIGNED_IR = "sub nuw"
    USER_FACING_NAME = "__builtin_subtract"


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


class CompareExpression(BasicIntegerExpression):
    @staticmethod
    def get_result_type(arguments: list[TypedExpression]) -> Type:
        return BoolType()


class IsEqualExpression(CompareExpression):
    SIGNED_IR = "icmp eq"
    UNSIGNED_IR = "icmp eq"
    USER_FACING_NAME = "__builtin_is_equal"


class IsGreaterThanExpression(CompareExpression):
    SIGNED_IR = "icmp sgt"
    UNSIGNED_IR = "icmp ugt"
    USER_FACING_NAME = "__builtin_is_greater_than"


class IsLessThanExpression(CompareExpression):
    SIGNED_IR = "icmp slt"
    UNSIGNED_IR = "icmp ult"
    USER_FACING_NAME = "__builtin_is_less_than"


class AlignOfExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
    ) -> None:
        assert len(arguments) == 0
        assert len(specialization) == 1
        (self._argument_type,) = specialization
        assert isinstance(self._argument_type, Type)

        self._result = ConstantExpression(
            SizeType(), str(self._argument_type.alignment)
        )

        TypedExpression.__init__(self, self._result.underlying_type, False)

    def __repr__(self) -> str:
        return f"AlignOf({self._argument_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return self._result.generate_ir(reg_gen)

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self._result.ir_ref_without_type_annotation

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot assign to `__builtin_alignof<...>()`")


class SizeOfExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
    ) -> None:
        assert len(arguments) == 0
        assert len(specialization) == 1
        (self._argument_type,) = specialization
        assert isinstance(self._argument_type, Type)

        self._result = ConstantExpression(SizeType(), str(self._argument_type.size))

        TypedExpression.__init__(self, self._result.underlying_type, False)

    def __repr__(self) -> str:
        return f"SizeOf({self._argument_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return self._result.generate_ir(reg_gen)

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self._result.ir_ref_without_type_annotation

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot assign to `__builtin_sizeof<...>()`")


class NarrowExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
    ) -> None:
        (self._argument,) = arguments
        (return_type,) = specialization
        assert isinstance(return_type, Type)

        self._arg_value_type = self._argument.underlying_type.convert_to_value_type()

        to_definition = return_type.definition
        from_definition = self._arg_value_type.definition
        assert isinstance(to_definition, IntegerDefinition)
        assert isinstance(from_definition, IntegerDefinition)

        assert from_definition.bits > to_definition.bits
        assert from_definition.is_signed == to_definition.is_signed

        TypedExpression.__init__(self, return_type, False)

    def __repr__(self) -> str:
        return f"Narrow({self._argument} to {self.underlying_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        conv_arg, extra_exprs_arg = do_implicit_conversion(
            self._argument, self._arg_value_type
        )
        ir_lines: list[str] = self.expand_ir(extra_exprs_arg, reg_gen)

        self.result_reg = next(reg_gen)
        return [
            *ir_lines,
            f"%{self.result_reg} = trunc {conv_arg.ir_ref_with_type_annotation}"
            f" to {self.underlying_type.ir_type}",
        ]

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot assign to `__builtin_narrow<...>(...)`")


class PtrToIntExpression(BuiltinCallable):
    def __init__(
        self, specialization: list[SpecializationItem], arguments: list[TypedExpression]
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) == 1

        # We don't attempt to dereference this at all. src_expr shouldn't have
        # more than one layer of indirection.
        (self._src_expr,) = arguments
        assert self._src_expr.has_address

        TypedExpression.__init__(self, IPtrType(), False)

    def __repr__(self) -> str:
        return f"PtrToInt({self._src_expr} to {self.underlying_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # Implicit conversions do not apply.

        self.result_reg = next(reg_gen)

        # <result> = ptrtoint <ty> <value> to <ty2>
        return [
            f"%{self.result_reg} = ptrtoint {self._src_expr.ir_ref_with_type_annotation}"
            f" to {self.underlying_type.ir_type}"
        ]

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_ptr_to_int()`")


class IntToPtrExpression(BuiltinCallable):
    def __init__(
        self, specialization: list[SpecializationItem], arguments: list[TypedExpression]
    ) -> None:
        assert len(specialization) == 1
        assert len(arguments) == 1

        (self._ptr_type,) = specialization
        assert isinstance(self._ptr_type, Type)
        assert self._ptr_type.is_reference

        (self._src_expr,) = arguments
        assert isinstance(self._src_expr.underlying_type, GenericIntType)

        TypedExpression.__init__(self, self._ptr_type.convert_to_value_type(), True)

    def __repr__(self) -> str:
        return f"IntToPtr({self._src_expr} to {self.underlying_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # Remove any indirect references
        conv_src_expr, extra_exprs = do_implicit_conversion(
            self._src_expr, self._src_expr.underlying_type
        )

        ir = self.expand_ir(extra_exprs, reg_gen)

        self.result_reg = next(reg_gen)

        # <result> = inttoptr <ty> <value> to <ty2>
        return ir + [
            f"%{self.result_reg} = inttoptr {conv_src_expr.ir_ref_with_type_annotation}"
            f" to {self.ir_type_annotation}",
        ]

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        pass


class BitcastExpression(BuiltinCallable):
    def __init__(
        self, specialization: list[SpecializationItem], arguments: list[TypedExpression]
    ) -> None:
        assert len(specialization) == 1
        assert len(arguments) == 1

        self.result_ref = None

        (self._src_expr,) = arguments
        (self._target_type,) = specialization
        assert isinstance(self._target_type, Type)

        src_type = self._src_expr.underlying_type

        # We can bitcast from ptr->ptr or non-aggregate->non-aggregate
        assert self._target_type.is_reference == src_type.is_reference

        # TODO: support floats, support references
        assert isinstance(self._target_type, GenericIntType)
        assert isinstance(src_type, GenericIntType)

        assert self._target_type.size == src_type.size

        TypedExpression.__init__(
            self,
            self._target_type.convert_to_value_type(),
            self._target_type.is_reference,
        )

    def __repr__(self) -> str:
        return f"Bitcast({self._src_expr} to {self.underlying_type})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # Remove any indirect references
        conv_src_expr, extra_exprs = do_implicit_conversion(
            self._src_expr, self._src_expr.underlying_type
        )

        ir = self.expand_ir(extra_exprs, reg_gen)

        if self.ir_type_annotation == conv_src_expr.ir_type_annotation:
            # Type is already correct, nothing to do
            self.result_ref = conv_src_expr.ir_ref_without_type_annotation
            return ir

        self.result_ref = f"%{next(reg_gen)}"

        # <result> = bitcast <ty> <value> to <ty2>
        return ir + [
            f"{self.result_ref} = bitcast {conv_src_expr.ir_ref_with_type_annotation}"
            f" to {self.ir_type_annotation}",
        ]

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_ref is not None
        return self.result_ref

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        self._src_expr.assert_can_write_to()


def get_builtin_callables() -> dict[str, type[BuiltinCallable]]:
    def get_integer_builtin(
        expression_class: PyType[BasicIntegerExpression | UnaryIntegerExpression],
    ):
        return (expression_class.USER_FACING_NAME, expression_class)

    integer_instructions = dict(
        map(
            get_integer_builtin,
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
                # NOTE: I haven't added the redundant compare equal/ not equal instructions here
                IsEqualExpression,
                IsGreaterThanExpression,
                IsLessThanExpression,
                UnaryMinusExpression,
                BitwiseNotExpression,
            ],
        )
    )

    return {
        **integer_instructions,
        "__builtin_alignof": AlignOfExpression,
        "__builtin_bitcast": BitcastExpression,
        "__builtin_int_to_ptr": IntToPtrExpression,
        "__builtin_narrow": NarrowExpression,
        "__builtin_ptr_to_int": PtrToIntExpression,
        "__builtin_sizeof": SizeOfExpression,
    }

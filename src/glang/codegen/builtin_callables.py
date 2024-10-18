from abc import abstractmethod
from typing import override

from glang.codegen.builtin_types import (
    BoolDefinition,
    BoolType,
    GenericIntType,
    HeapArrayDefinition,
    IEEEFloatDefinition,
    IntegerDefinition,
    IPtrType,
    SizeType,
    StackArrayDefinition,
    VoidType,
    format_array_dims_for_ir,
)
from glang.codegen.expressions import ConstantExpression, StaticTypedExpression
from glang.codegen.interfaces import (
    IRContext,
    IROutput,
    SpecializationItem,
    Type,
    TypedExpression,
)
from glang.codegen.type_conversions import do_implicit_conversion
from glang.codegen.user_facing_errors import OperandError
from glang.parser.lexer_parser import Meta

NUMERIC_TYPES = (IntegerDefinition, BoolDefinition, IEEEFloatDefinition)


class BuiltinCallable(StaticTypedExpression):
    pass


class UnaryExpression(BuiltinCallable):
    IR_FORMAT_STR = ""
    USER_FACING_NAME = ""
    EXPECTED_TYPES = ()

    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        (self._arg,) = arguments

        assert self.IR_FORMAT_STR is not None
        assert len(specialization) == 0

        definition = self._arg.result_type.definition
        assert isinstance(definition, self.EXPECTED_TYPES)

        self._arg_type = self._arg.result_type.convert_to_value_type()
        super().__init__(self._arg_type, Type.Kind.VALUE, meta)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._arg})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#add-instruction (and below)
        conv, extra_exprs = do_implicit_conversion(self._arg, self._arg_type)

        ir_output = IROutput()
        ir_output.extend(self.expand_ir(extra_exprs, ctx))

        self.result_reg = next(ctx.reg_gen)

        op = self.IR_FORMAT_STR.format_map(
            {
                "type": self._arg_type.ir_type,
                "arg": conv.ir_ref_without_type_annotation,
            }
        )

        dbg = self.add_di_location(ctx, ir_output)

        ir_output.lines.append(f"%{self.result_reg} = {op}, {dbg}")
        return ir_output

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError(f"cannot assign to `{self.USER_FACING_NAME}(..., ...)`")


class BitwiseNotExpression(UnaryExpression):
    EXPECTED_TYPES = (IntegerDefinition, BoolDefinition)
    USER_FACING_NAME = "__builtin_bitwise_not"
    IR_FORMAT_STR = "xor {type} -1, {arg}"


class UnaryIntegerMinusExpression(UnaryExpression):
    EXPECTED_TYPES = IntegerDefinition
    USER_FACING_NAME = "__builtin_minus"
    IR_FORMAT_STR = "sub {type} 0, {arg}"


class UnaryFloatingMinusExpression(UnaryExpression):
    EXPECTED_TYPES = IEEEFloatDefinition
    USER_FACING_NAME = "__builtin_fminus"
    IR_FORMAT_STR = "fneg {type} {arg}"


class BasicNumericExpression(BuiltinCallable):
    FLOATING_POINT_IR = None
    SIGNED_IR = None
    UNSIGNED_IR = None
    USER_FACING_NAME = None

    @staticmethod
    @abstractmethod
    def get_result_type(arguments: list[TypedExpression]) -> Type:
        pass

    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        lhs, rhs = arguments
        # This is not a user-facing function, we don't need sensible error messages
        assert self.USER_FACING_NAME is not None

        assert len(specialization) == 0
        lhs_definition = lhs.result_type.definition
        rhs_definition = rhs.result_type.definition
        assert isinstance(lhs_definition, NUMERIC_TYPES)
        assert isinstance(rhs_definition, NUMERIC_TYPES)
        assert lhs.result_type == rhs.result_type

        super().__init__(self.get_result_type(arguments), Type.Kind.VALUE, meta)

        self._arg_type = lhs.result_type.convert_to_value_type()
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._lhs}, {self._rhs})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#add-instruction (and below)
        conv_lhs, extra_exprs_lhs = do_implicit_conversion(self._lhs, self._arg_type)
        conv_rhs, extra_exprs_rhs = do_implicit_conversion(self._rhs, self._arg_type)

        ir_output = IROutput()
        ir_output.extend(self.expand_ir(extra_exprs_lhs, ctx))
        ir_output.extend(self.expand_ir(extra_exprs_rhs, ctx))

        self.result_reg = ctx.next_reg()

        if isinstance(self._arg_type.definition, IEEEFloatDefinition):
            ir = self.FLOATING_POINT_IR
        elif isinstance(self._arg_type.definition, IntegerDefinition):
            ir = (
                self.SIGNED_IR
                if self._arg_type.definition.is_signed
                else self.UNSIGNED_IR
            )
        else:
            assert isinstance(self._arg_type.definition, BoolDefinition)
            ir = self.UNSIGNED_IR

        assert ir is not None
        dbg = self.add_di_location(ctx, ir_output)

        # eg. for addition
        # <result> = add [nuw] [nsw] <ty> <op1>, <op2>  ; yields ty:result
        ir_output.lines.append(
            f"%{self.result_reg} = {ir} {conv_lhs.ir_ref_with_type_annotation}, "
            f"{conv_rhs.ir_ref_without_type_annotation}, {dbg}"
        )

        return ir_output

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError(f"cannot assign to `{self.USER_FACING_NAME}(..., ...)`")


class ArithmeticExpression(BasicNumericExpression):
    @staticmethod
    def get_result_type(arguments: list[TypedExpression]) -> Type:
        return arguments[0].result_type.convert_to_value_type()


class AddExpression(ArithmeticExpression):
    FLOATING_POINT_IR = "fadd"
    SIGNED_IR = "add nsw"
    UNSIGNED_IR = "add nuw"
    USER_FACING_NAME = "__builtin_add"


class SubExpression(ArithmeticExpression):
    FLOATING_POINT_IR = "fsub"
    SIGNED_IR = "sub nsw"
    UNSIGNED_IR = "sub nuw"
    USER_FACING_NAME = "__builtin_subtract"


class MultiplyExpression(ArithmeticExpression):
    FLOATING_POINT_IR = "fmul"
    SIGNED_IR = "mul nsw"
    UNSIGNED_IR = "mul nuw"
    USER_FACING_NAME = "__builtin_multiply"


class DivideExpression(ArithmeticExpression):
    FLOATING_POINT_IR = "fdiv"
    SIGNED_IR = "sdiv"
    UNSIGNED_IR = "udiv"
    USER_FACING_NAME = "__builtin_divide"


class RemainderExpression(ArithmeticExpression):
    FLOATING_POINT_IR = "frem"
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


class CompareExpression(BasicNumericExpression):
    @staticmethod
    def get_result_type(arguments: list[TypedExpression]) -> Type:
        return BoolType()


class IsEqualExpression(CompareExpression):
    FLOATING_POINT_IR = "fcmp ueq"  # Ordered and equal (can we relax this?)
    SIGNED_IR = "icmp eq"
    UNSIGNED_IR = "icmp eq"
    USER_FACING_NAME = "__builtin_is_equal"


class IsGreaterThanExpression(CompareExpression):
    FLOATING_POINT_IR = "fcmp ugt"
    SIGNED_IR = "icmp sgt"
    UNSIGNED_IR = "icmp ugt"
    USER_FACING_NAME = "__builtin_is_greater_than"


class IsLessThanExpression(CompareExpression):
    FLOATING_POINT_IR = "fcmp ult"
    SIGNED_IR = "icmp slt"
    UNSIGNED_IR = "icmp ult"
    USER_FACING_NAME = "__builtin_is_less_than"


class AlignOfExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(arguments) == 0
        assert len(specialization) == 1
        (self._argument_type,) = specialization
        assert isinstance(self._argument_type, Type)

        self._result = ConstantExpression(
            SizeType(), str(self._argument_type.alignment), meta
        )

        super().__init__(self._result.underlying_type, Type.Kind.VALUE, meta)

    def __repr__(self) -> str:
        return f"AlignOf({self._argument_type})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        return self._result.generate_ir(ctx)

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
        meta: Meta,
    ) -> None:
        assert len(arguments) == 0
        assert len(specialization) == 1
        (self._argument_type,) = specialization
        assert isinstance(self._argument_type, Type)

        self._result = ConstantExpression(
            SizeType(), str(self._argument_type.size), meta
        )

        super().__init__(self._result.underlying_type, Type.Kind.VALUE, meta)

    def __repr__(self) -> str:
        return f"SizeOf({self._argument_type})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        return self._result.generate_ir(ctx)

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
        meta: Meta,
    ) -> None:
        (self._argument,) = arguments
        (return_type,) = specialization
        assert isinstance(return_type, Type)

        self._arg_value_type = self._argument.result_type.convert_to_value_type()

        to_definition = return_type.definition
        from_definition = self._arg_value_type.definition
        assert isinstance(to_definition, IntegerDefinition)
        assert isinstance(from_definition, IntegerDefinition)

        assert from_definition.bits > to_definition.bits
        assert from_definition.is_signed == to_definition.is_signed

        super().__init__(return_type, Type.Kind.VALUE, meta)

    def __repr__(self) -> str:
        return f"Narrow({self._argument} to {self.underlying_type})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        conv_arg, extra_exprs_arg = do_implicit_conversion(
            self._argument, self._arg_value_type
        )
        ir_output = self.expand_ir(extra_exprs_arg, ctx)

        dbg = self.add_di_location(ctx, ir_output)

        self.result_reg = ctx.next_reg()

        ir_output.lines.append(
            f"%{self.result_reg} = trunc {conv_arg.ir_ref_with_type_annotation}"
            f" to {self.underlying_type.ir_type}, {dbg}"
        )

        return ir_output

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot assign to `__builtin_narrow<...>(...)`")


class PtrToIntExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) == 1

        # We don't attempt to dereference this at all. src_expr shouldn't have
        # more than one layer of indirection.
        (self._src_expr,) = arguments
        assert self._src_expr.has_address

        super().__init__(IPtrType(), Type.Kind.VALUE, meta)

    def __repr__(self) -> str:
        return f"PtrToInt({self._src_expr} to {self.underlying_type})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # Implicit conversions do not apply.
        ir_output = IROutput()

        self.result_reg = ctx.next_reg()

        dbg = self.add_di_location(ctx, ir_output)

        # <result> = ptrtoint <ty> <value> to <ty2>
        ir_output.lines.append(
            f"%{self.result_reg} = ptrtoint {self._src_expr.ir_ref_with_type_annotation}"
            f" to {self.underlying_type.ir_type}, {dbg}"
        )

        return ir_output

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_ptr_to_int()`")


class IntToPtrExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 1
        assert len(arguments) == 1

        (self._ptr_type,) = specialization
        assert isinstance(self._ptr_type, Type)
        assert self._ptr_type.storage_kind.is_reference()

        (self._src_expr,) = arguments
        assert isinstance(self._src_expr.result_type, GenericIntType)

        super().__init__(
            self._ptr_type.convert_to_value_type(),
            self._ptr_type.storage_kind,
            meta,
        )

    def __repr__(self) -> str:
        return f"IntToPtr({self._src_expr} to {self.underlying_type})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # Remove any indirect references
        conv_src_expr, extra_exprs = do_implicit_conversion(
            self._src_expr, self._src_expr.result_type
        )

        ir = self.expand_ir(extra_exprs, ctx)

        self.result_reg = ctx.next_reg()

        dbg = self.add_di_location(ctx, ir)

        # <result> = inttoptr <ty> <value> to <ty2>
        ir.lines.append(
            f"%{self.result_reg} = inttoptr {conv_src_expr.ir_ref_with_type_annotation}"
            f" to {self.ir_type_annotation}, {dbg}"
        )

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        pass


class BitcastExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 1
        assert len(arguments) == 1

        self.result_ref = None

        (self._src_expr,) = arguments
        (self._target_type,) = specialization
        assert isinstance(self._target_type, Type)

        src_type = self._src_expr.result_type

        # We can bitcast from ptr->ptr or non-aggregate->non-aggregate
        #  We can also bitcast away const
        assert (
            self._target_type.storage_kind.is_reference()
            == src_type.storage_kind.is_reference()
        )

        # TODO: support floats, support references
        assert isinstance(self._target_type, GenericIntType)
        assert isinstance(src_type, GenericIntType)

        assert self._target_type.size == src_type.size

        super().__init__(
            self._target_type.convert_to_value_type(),
            self._target_type.storage_kind,
            meta,
        )

    def __repr__(self) -> str:
        return f"Bitcast({self._src_expr} to {self.underlying_type})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # Remove any indirect references
        conv_src_expr, extra_exprs = do_implicit_conversion(
            self._src_expr, self._src_expr.result_type
        )

        ir = self.expand_ir(extra_exprs, ctx)

        if self.ir_type_annotation == conv_src_expr.ir_type_annotation:
            # Type is already correct, nothing to do
            self.result_ref = conv_src_expr.ir_ref_without_type_annotation
            return ir

        self.result_ref = f"%{ctx.next_reg()}"

        dbg = self.add_di_location(ctx, ir)

        # <result> = bitcast <ty> <value> to <ty2>
        ir.lines.append(
            f"{self.result_ref} = bitcast {conv_src_expr.ir_ref_with_type_annotation}"
            f" to {self.ir_type_annotation}, {dbg}",
        )

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_ref is not None
        return self.result_ref

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        self._src_expr.assert_can_write_to()


class ArrayIndexExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) >= 2

        array_ptr, *indices = arguments

        assert array_ptr.has_address

        self._type_of_array: Type = array_ptr.result_type.convert_to_value_type()
        self._array_ptr = array_ptr

        array_definition = self._type_of_array.definition
        assert isinstance(array_definition, StackArrayDefinition | HeapArrayDefinition)

        if isinstance(array_definition, StackArrayDefinition):
            assert len(array_definition.dimensions) == len(indices)

        else:
            assert len(indices) == 1 + len(array_definition.known_dimensions)

        self._element_type: Type = array_definition.member
        self._conversion_exprs: list[TypedExpression] = []

        # Now convert all the indices into integers using standard implicit rules
        self._indices: list[TypedExpression] = []
        for index in indices:
            index_expr, conversions = do_implicit_conversion(
                index, SizeType(), "array index access"
            )
            self._indices.append(index_expr)
            self._conversion_exprs.extend(conversions)

        result_type = array_ptr.result_type_as_if_borrowed
        super().__init__(
            self._element_type.convert_to_value_type(),
            result_type.storage_kind,
            meta,
        )

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction
        array_def = self._type_of_array.definition
        assert isinstance(array_def, StackArrayDefinition | HeapArrayDefinition)

        ir = self.expand_ir(self._conversion_exprs, ctx)

        # To access a stack array, we need to dereference the pointer returned
        # by `alloca` to get the address of the first element, and then we can
        # index into it. For heap arrays, we already have its address so we can
        # index it immediately.
        if isinstance(array_def, StackArrayDefinition):
            assert self.meta is not None
            indices_ir = [
                ConstantExpression(
                    SizeType(), "0", self.meta
                ).ir_ref_with_type_annotation
            ]
            gep_type_ir = self._type_of_array.ir_type
        else:
            indices_ir = []
            gep_type_ir = format_array_dims_for_ir(
                array_def.known_dimensions, array_def.member
            )

        for index in self._indices:
            indices_ir.append(index.ir_ref_with_type_annotation)

        self.result_reg = ctx.next_reg()
        dbg = self.add_di_location(ctx, ir)

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        ir.lines.append(
            f"%{self.result_reg} = getelementptr inbounds {gep_type_ir},"
            f" {self._array_ptr.ir_ref_with_type_annotation}, {', '.join(indices_ir)},"
            f" {dbg}",
        )

        if self._element_type.storage_kind.is_reference():
            self.result_reg = self.dereference_double_indirection(ctx, ir)

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        indices = ", ".join(map(repr, self._indices))
        return f"ArrayIndexExpression({self._array_ptr}[{indices}])"

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the individual members are initialized?
        self._array_ptr.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._array_ptr.assert_can_write_to()


class TrapExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) == 0
        StaticTypedExpression.__init__(
            self,
            VoidType(),
            Type.Kind.VALUE,
            meta,
        )

    def __repr__(self) -> str:
        return "TrapExpression()"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)
        ir.lines.append(f"call void @llvm.trap(), {dbg}")
        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert False

    def assert_can_read_from(self) -> None:
        assert False

    def assert_can_write_to(self) -> None:
        assert False


class UnreachableExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) == 0
        StaticTypedExpression.__init__(
            self,
            VoidType(),
            Type.Kind.VALUE,
            meta,
        )

    def __repr__(self) -> str:
        return "UnreachableExpression()"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)
        ir.lines.append(f"unreachable, {dbg}")
        return ir

    @override
    def is_return_guaranteed(self) -> bool:
        return True

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert False

    def assert_can_read_from(self) -> None:
        assert False

    def assert_can_write_to(self) -> None:
        assert False


class VolatileWriteExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) == 2

        (self._addr, self._to_write) = arguments
        assert self._addr.result_type.storage_kind.is_reference()
        assert not self._to_write.result_type.storage_kind.is_reference()

        assert (
            self._addr.result_type.convert_to_value_type() == self._to_write.result_type
        )

        super().__init__(
            VoidType(),
            Type.Kind.VALUE,
            meta,
        )

    def __repr__(self) -> str:
        return f"VolatileWriteExpression({self._to_write} to {self._addr})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # Remove any indirect references
        conv_to_write, extra_exprs_to_write = do_implicit_conversion(
            self._to_write, self._to_write.result_type
        )

        conv_addr, extra_exprs_addr = do_implicit_conversion(
            self._addr, self._addr.result_type
        )

        ir = IROutput()
        ir.extend(self.expand_ir(extra_exprs_to_write, ctx))
        ir.extend(self.expand_ir(extra_exprs_addr, ctx))

        dbg = self.add_di_location(ctx, ir)

        # store volatile <ty> <value>, ptr <pointer>
        ir.lines.append(
            f"store volatile {conv_to_write.ir_ref_with_type_annotation}, "
            f" {conv_addr.ir_ref_with_type_annotation}, {dbg}",
        )
        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert False

    def assert_can_read_from(self) -> None:
        raise OperandError("Cannot read from `__builtin_volatile_write()`")

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_volatile_write()`")


class VolatileReadExpression(BuiltinCallable):
    def __init__(
        self,
        specialization: list[SpecializationItem],
        arguments: list[TypedExpression],
        meta: Meta,
    ) -> None:
        assert len(specialization) == 0
        assert len(arguments) == 1

        self.result_ref = None

        (self._addr,) = arguments
        assert self._addr.result_type.storage_kind.is_reference()

        super().__init__(
            self._addr.result_type.convert_to_value_type(),
            Type.Kind.VALUE,
            meta,
        )

    def __repr__(self) -> str:
        return f"VolatileReadExpression({self._to_write} to {self._addr})"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # Remove any indirect references
        conv_addr, extra_exprs_addr = do_implicit_conversion(
            self._addr, self._addr.result_type
        )

        ir = self.expand_ir(extra_exprs_addr, ctx)

        self.result_ref = f"%{ctx.next_reg()}"

        dbg = self.add_di_location(ctx, ir)

        # <result> = load volatile <ty> <value>, ptr <pointer>
        ir.lines.append(
            f"{self.result_ref} = load volatile {self.result_type.ir_type}, "
            f"{conv_addr.ir_ref_with_type_annotation}, {dbg}",
        )
        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_ref is not None
        return self.result_ref

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("Cannot assign to `__builtin_volatile_write()`")


def get_builtin_callables() -> dict[str, type[BuiltinCallable]]:
    def get_arithmetic_builtin(
        expression_class: type[BasicNumericExpression | UnaryExpression],
    ):
        assert expression_class.USER_FACING_NAME is not None
        return (expression_class.USER_FACING_NAME, expression_class)

    integer_instructions = dict(
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
                # NOTE: I haven't added the redundant compare equal/ not equal instructions here
                IsEqualExpression,
                IsGreaterThanExpression,
                IsLessThanExpression,
                UnaryIntegerMinusExpression,
                UnaryFloatingMinusExpression,
                BitwiseNotExpression,
            ],
        )
    )

    return {
        **integer_instructions,
        "__builtin_alignof": AlignOfExpression,
        "__builtin_array_index": ArrayIndexExpression,
        "__builtin_bitcast": BitcastExpression,
        "__builtin_int_to_ptr": IntToPtrExpression,
        "__builtin_narrow": NarrowExpression,
        "__builtin_ptr_to_int": PtrToIntExpression,
        "__builtin_sizeof": SizeOfExpression,
        "__builtin_trap": TrapExpression,
        "__builtin_unreachable": UnreachableExpression,
        "__builtin_volatile_read": VolatileReadExpression,
        "__builtin_volatile_write": VolatileWriteExpression,
    }

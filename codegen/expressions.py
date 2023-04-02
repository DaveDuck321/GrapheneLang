from functools import cached_property
from typing import Iterator

from .builtin_types import (
    ArrayDefinition,
    FunctionSignature,
    IntType,
    SizeType,
    StructDefinition,
)
from .interfaces import Type, TypedExpression, Variable
from .type_conversions import (
    assert_is_implicitly_convertible,
    do_implicit_conversion,
)
from .user_facing_errors import (
    ArrayIndexCount,
    BorrowTypeError,
    DoubleBorrowError,
    OperandError,
    TypeCheckerError,
)


class ConstantExpression(TypedExpression):
    def __init__(self, cst_type: Type, value: str) -> None:
        super().__init__(cst_type, False)

        self.value = cst_type.graphene_literal_to_ir_constant(value)

    def __repr__(self) -> str:
        return f"ConstantExpression({self.underlying_type}, {self.value})"

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self.value

    def assert_can_read_from(self) -> None:
        # Can always read the result of a constant expression.
        pass

    def assert_can_write_to(self) -> None:
        # Can never write to a constant expression (an rvalue).
        raise OperandError(f"cannot modify the constant {self.value}")


class VariableReference(TypedExpression):
    def __init__(self, variable: Variable) -> None:
        # A variable with a reference type needs borrowing before it becomes a true reference
        super().__init__(
            variable.type.convert_to_value_type(),
            True,
            variable.type.is_borrowed_reference,
        )

        self.variable = variable

    def __repr__(self) -> str:
        return (
            f"VariableReference({self.variable.user_facing_name}: {self.variable.type})"
        )

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self._ir_ref

    def assert_can_read_from(self) -> None:
        self.variable.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self.variable.assert_can_write_to()

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        self._ir_ref = self.variable.ir_ref_without_type_annotation

        ir = []
        if self.variable.type.is_borrowed_reference:
            self._ir_ref = f"%{self.dereference_double_indirection(reg_gen, ir)}"

        return ir


class FunctionParameter(TypedExpression):
    def __init__(self, expr_type: Type) -> None:
        super().__init__(expr_type, False)

    def __repr__(self) -> str:
        return f"FunctionParameter({self.underlying_type})"

    def set_reg(self, reg: int) -> None:
        self.result_reg = reg

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_reg is not None
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        # We should only write to the implicit stack variable
        #   Writing directly to a parameter is a codegen error
        assert False


class FunctionCallExpression(TypedExpression):
    def __init__(
        self, signature: FunctionSignature, args: list[TypedExpression]
    ) -> None:
        if signature.return_type.is_borrowed_reference:
            # The user needs to borrow again if they want the actual reference
            super().__init__(signature.return_type.convert_to_value_type(), True, True)
        else:
            # The function returns a value (not an address), so we can't later borrow it
            super().__init__(signature.return_type, False)

        for arg, arg_type in zip(args, signature.arguments, strict=True):
            arg.assert_can_read_from()
            # We do check this during overload resolution, but you can never be
            # too careful.
            assert_is_implicitly_convertible(arg, arg_type, "function call")

        self.signature = signature
        self.args = args

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#call-instruction

        ir_lines: list[str] = []
        conv_args: list[TypedExpression] = []

        for arg, arg_type in zip(self.args, self.signature.arguments, strict=True):
            conv_arg, extra_exprs = do_implicit_conversion(arg, arg_type)

            ir_lines += self.expand_ir(extra_exprs, reg_gen)
            conv_args.append(conv_arg)

        args_ir = map(lambda arg: arg.ir_ref_with_type_annotation, conv_args)

        call_expr = f"call {self.signature.ir_ref}({str.join(', ', args_ir)})"

        # We cannot assign `void` to a register.
        if not self.signature.return_type.is_void:
            self.result_reg = next(reg_gen)
            call_expr = f"%{self.result_reg} = {call_expr}"

        ir_lines.append(call_expr)

        return ir_lines

    def __repr__(self) -> str:
        return f"FunctionCallExpression({self.signature})"

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        # Can read any return type. Let the caller check if it's compatible.
        pass

    def assert_can_write_to(self) -> None:
        # TODO: Maybe the error message could be better? Atm we have:
        #   Error: cannot assign to non-reference 'int' since it does not have an address
        pass


class BorrowExpression(TypedExpression):
    def __init__(self, expr: TypedExpression) -> None:
        self._expr = expr

        if expr.underlying_type.is_borrowed_reference:
            raise DoubleBorrowError(expr.underlying_type.get_user_facing_name(True))

        if not expr.is_indirect_pointer_to_type:
            raise BorrowTypeError(expr.underlying_type.get_user_facing_name(True))

        super().__init__(expr.underlying_type.take_reference(), False)

    def __repr__(self) -> str:
        return f"Borrow({repr(self._expr)})"

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self._expr.ir_ref_without_type_annotation

    def assert_can_read_from(self) -> None:
        self._expr.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._expr.assert_can_write_to()


class StructMemberAccess(TypedExpression):
    def __init__(self, lhs: TypedExpression, member_name: str) -> None:
        self._member_name = member_name
        self._lhs = lhs

        self._struct_type = lhs.underlying_type
        underlying_definition = lhs.underlying_type.definition
        if not isinstance(underlying_definition, StructDefinition):
            raise TypeCheckerError(
                "struct member access",
                lhs.underlying_type.get_user_facing_name(False),
                "{...}",
            )

        self._index, self._member_type = underlying_definition.get_member_by_name(
            member_name
        )

        # If the member is a reference we can always calculate an address
        if self._member_type.is_borrowed_reference:
            super().__init__(self._member_type.convert_to_value_type(), True)
        else:
            # We only know an address if the struct itself has an address
            super().__init__(self._member_type, lhs.has_address)

    def generate_ir_from_known_address(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction

        # In llvm structs behind a pointer are treated like an array
        pointer_offset = ConstantExpression(IntType(), "0")
        index = ConstantExpression(IntType(), str(self._index))

        self.result_reg = next(reg_gen)

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        ir = [
            f"%{self.result_reg} = getelementptr inbounds {self._struct_type.ir_type},"
            f" {self._lhs.ir_ref_with_type_annotation}, "
            f"{pointer_offset.ir_ref_with_type_annotation}, {index.ir_ref_with_type_annotation}",
        ]

        # Prevent double indirection, dereference the element pointer to get the underlying reference
        if self._member_type.is_borrowed_reference:
            self.result_reg = self.dereference_double_indirection(reg_gen, ir)

        return ir

    def generate_ir_without_known_address(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#insertvalue-instruction

        # <result> = extractvalue <aggregate type> <val>, <idx>{, <idx>}*
        self.result_reg = next(reg_gen)
        return [
            f"%{self.result_reg} = extractvalue {self._lhs.ir_ref_with_type_annotation},"
            f" {self._index}"
        ]

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        if self._lhs.has_address:
            return self.generate_ir_from_known_address(reg_gen)
        else:
            return self.generate_ir_without_known_address(reg_gen)

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return (
            f"StructMemberAccess({self.underlying_type.get_user_facing_name(False)}"
            f".{self._member_name}: {self.underlying_type})"
        )

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the members are initialized?
        pass

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        if not self.has_address:
            raise OperandError("cannot modify temporary struct")

        # TODO: check if the reference is const


class ArrayIndexAccess(TypedExpression):
    def __init__(
        self, array_ptr: TypedExpression, indices: list[TypedExpression]
    ) -> None:
        # NOTE: needs address since getelementptr must be used for runtime indexing
        assert array_ptr.has_address

        self._type_of_array: Type = array_ptr.underlying_type
        self._array_ptr = array_ptr

        array_definition = self._type_of_array.definition
        if not isinstance(array_definition, ArrayDefinition):
            raise TypeCheckerError(
                "array index access",
                array_ptr.underlying_type.get_user_facing_name(False),
                "T[...]",
            )

        if len(array_definition._dimensions) != len(indices):
            raise ArrayIndexCount(
                self._type_of_array.get_user_facing_name(False),
                len(indices),
                len(array_definition._dimensions),
            )

        self._element_type: Type = array_definition._element_type
        self._conversion_exprs: list[TypedExpression] = []

        # Now convert all the indices into integers using standard implicit rules
        self._indices: list[TypedExpression] = []
        for index in indices:
            index_expr, conversions = do_implicit_conversion(
                index, SizeType(), "array index access"
            )
            self._indices.append(index_expr)
            self._conversion_exprs.extend(conversions)

        super().__init__(self._element_type.convert_to_value_type(), True)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction
        conversion_ir = self.expand_ir(self._conversion_exprs, reg_gen)

        pointer_offset = ConstantExpression(IntType(), "0")
        indices_ir = [pointer_offset.ir_ref_with_type_annotation]
        for index in self._indices:
            indices_ir.append(index.ir_ref_with_type_annotation)

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        self.result_reg = next(reg_gen)
        ir = [
            *conversion_ir,
            f"%{self.result_reg} = getelementptr inbounds {self._type_of_array.ir_type},"
            f" {self._array_ptr.ir_ref_with_type_annotation}, {', '.join(indices_ir)}",
        ]

        if self._element_type.is_borrowed_reference:
            self.result_reg = self.dereference_double_indirection(reg_gen, ir)

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        indices = ", ".join(map(repr, self._indices))
        return f"ArrayIndexAccess({self._array_ptr}[{indices}])"

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the members are initialized?
        pass

    def assert_can_write_to(self) -> None:
        pass

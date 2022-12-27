from functools import cached_property
from typing import Iterator

from .builtin_types import FunctionSignature, IntType, StructDefinition, ArrayDefinition
from .generatable import StackVariable
from .interfaces import Type, TypedExpression, Variable
from .type_conversions import (
    assert_is_implicitly_convertible,
    dereference_to_single_reference,
    do_implicit_conversion,
    Decay,
)
from .user_facing_errors import (
    ArrayIndexCount,
    BorrowTypeError,
    OperandError,
    TypeCheckerError,
    assert_else_throw,
    throw,
)


class ConstantExpression(TypedExpression):
    def __init__(self, cst_type: Type, value: str) -> None:
        super().__init__(cst_type)

        self.value = cst_type.to_ir_constant(value)

    def __repr__(self) -> str:
        return f"ConstantExpression({self.type}, {self.value})"

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return self.value

    def assert_can_read_from(self) -> None:
        # Can always read the result of a constant expression.
        pass

    def assert_can_write_to(self) -> None:
        # Can never write to a constant expression (an rvalue).
        throw(OperandError(f"Cannot modify the constant {self.value}"))


class VariableReference(TypedExpression):
    def __init__(self, variable: Variable) -> None:
        super().__init__(variable.type.to_unborrowed_ref())

        self.variable = variable

    def __repr__(self) -> str:
        return (
            f"VariableReference({self.variable.user_facing_name}: {self.variable.type})"
        )

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return self.variable.ir_ref_without_type_annotation

    def set_initialized_through_mutable_borrow(self) -> None:
        assert isinstance(self.variable, StackVariable)
        self.variable.initialized = True

    def assert_can_read_from(self) -> None:
        # Can ready any initialized variable.
        assert isinstance(self.variable, StackVariable)
        assert_else_throw(
            self.variable.initialized,
            OperandError(
                f"Cannot use uninitialized variable '{self.variable.user_facing_name}'"
            ),
        )

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        assert_else_throw(
            not self.variable.constant,
            OperandError(
                f"Cannot modify constant variable '{self.variable.user_facing_name}'"
            ),
        )


class FunctionParameter(TypedExpression):
    def __init__(self, expr_type: Type) -> None:
        this_type = expr_type.copy()
        # Implicit borrow here.
        # XXX can't use to_borrowed(), since that increments ref_depth. We only
        # want to flag that the type has been borrowed.
        this_type.is_borrowed = expr_type.is_reference

        super().__init__(this_type)

    def __repr__(self) -> str:
        return f"FunctionParameter({self.type})"

    def set_reg(self, reg: int) -> None:
        self.result_reg = reg

    @cached_property
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
        super().__init__(signature.return_type)

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

        self.result_reg = next(reg_gen)
        args_ir = map(lambda arg: arg.ir_ref_with_type_annotation, conv_args)

        ir_lines.append(
            f"%{self.result_reg} = call {self.signature.ir_ref}({str.join(', ', args_ir)})"
        )

        return ir_lines

    def __repr__(self) -> str:
        return f"FunctionCallExpression({self.signature})"

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        # Can read any return type. Let the caller check if it's compatible.
        pass

    def assert_can_write_to(self) -> None:
        # Can write to any reference return type. TODO we don't have references
        # yet, so any attempt to write to the return value should fail for now.
        throw(OperandError(f"Cannot modify the value returned by {self.signature}"))


class Borrow(TypedExpression):
    def __init__(self, expr: TypedExpression) -> None:
        this_type = expr.type

        assert_else_throw(
            this_type.is_unborrowed_ref,
            BorrowTypeError(this_type.get_user_facing_name(False)),
        )

        # FIXME: const borrows should not initialize a variable
        if isinstance(expr, VariableReference):
            expr.set_initialized_through_mutable_borrow()

        self._expr = expr

        super().__init__(this_type.to_borrowed_ref())

    def __repr__(self) -> str:
        return f"Borrow({repr(self._expr)})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return self._expr.generate_ir(reg_gen)

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return self._expr.ir_ref_without_type_annotation

    def assert_can_read_from(self) -> None:
        self._expr.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._expr.assert_can_write_to()


class StructMemberAccess(TypedExpression):
    def __init__(self, lhs: TypedExpression, member_name: str) -> None:
        self._member_name = member_name

        self._deref_exprs = []
        self._lhs = lhs

        # TODO: this is kinda ugly now
        if lhs.type.is_pointer:
            if lhs.type.is_unborrowed_ref:
                self._lhs = Decay(lhs)

            self._deref_exprs.extend(dereference_to_single_reference(self._lhs))
            if len(self._deref_exprs) != 0:
                self._lhs = self._deref_exprs[-1]

        self._struct_type = self._lhs.type.to_value_type()

        struct_definition = self._struct_type.definition
        assert_else_throw(
            isinstance(struct_definition, StructDefinition),
            TypeCheckerError(
                "struct member access",
                self._struct_type.get_user_facing_name(False),
                "{...}",
            ),
        )
        assert isinstance(struct_definition, StructDefinition)

        self._access_index, member_type = struct_definition.get_member(member_name)

        # FIXME changed this to is_pointer, is that correct?
        self._source_struct_is_reference = lhs.type.is_pointer

        super().__init__(
            member_type.to_unborrowed_ref()
            if self._source_struct_is_reference
            else member_type
        )

    def generate_ir_for_reference_type(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction
        deref_ir = []
        for expression in self._deref_exprs:
            deref_ir.extend(expression.generate_ir(reg_gen))

        index = ConstantExpression(IntType(), str(self._access_index))

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        self.result_reg = next(reg_gen)
        return [
            *deref_ir,
            f"%{self.result_reg} = getelementptr inbounds {self._struct_type.ir_type},"
            f" {self._lhs.ir_ref_with_type_annotation}, {index.ir_ref_with_type_annotation}",
        ]

    def generate_ir_for_value_type(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#insertvalue-instruction
        assert len(self._deref_exprs) == 0

        # <result> = extractvalue <aggregate type> <val>, <idx>{, <idx>}*
        self.result_reg = next(reg_gen)
        return [
            f"%{self.result_reg} = extractvalue {self._lhs.ir_ref_with_type_annotation},"
            f" {self._access_index}"
        ]

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        if self._source_struct_is_reference:
            return self.generate_ir_for_reference_type(reg_gen)

        return self.generate_ir_for_value_type(reg_gen)

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return (
            f"StructMemberAccess({self._struct_type.get_user_facing_name(False)}"
            f".{self._member_name}: {self.type})"
        )

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the members are initialized?
        pass

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        assert_else_throw(
            not self._source_struct_is_reference,
            OperandError("Cannot modify temporary struct"),
        )
        if self._source_struct_is_reference:
            # TODO: check if the reference is const
            pass


class ArrayIndexAccess(TypedExpression):
    def __init__(
        self, array_ptr: TypedExpression, indices: list[TypedExpression]
    ) -> None:
        # NOTE: needs pointer since getelementptr must be used for runtime indexing
        assert array_ptr.type.is_pointer

        array_definition = array_ptr.type.definition
        assert_else_throw(
            isinstance(array_definition, ArrayDefinition),
            TypeCheckerError(
                "array index access",
                array_ptr.type.get_user_facing_name(False),
                "T[...]",
            ),
        )
        assert isinstance(array_definition, ArrayDefinition)
        assert_else_throw(
            len(array_definition._dimensions) == len(indices),
            ArrayIndexCount(
                array_ptr.type.get_user_facing_name(False),
                len(indices),
                len(array_definition._dimensions),
            ),
        )

        self._array_ptr = array_ptr
        self._conversion_exprs: list[TypedExpression] = []

        # We need to operate on a pure reference type so we can dereference
        if array_ptr.type.is_unborrowed_ref:
            self._array_ptr = Decay(array_ptr)

        # Recursively dereference (without any further implicit conversions)
        self._conversion_exprs.extend(dereference_to_single_reference(self._array_ptr))
        if len(self._conversion_exprs) != 0:
            self._array_ptr = self._conversion_exprs[-1]

        self._array_type = array_ptr.type.to_value_type()

        # Now convert all the indices into integers using standard implicit rules
        self._indices: list[TypedExpression] = []
        for index in indices:
            # FIXME: use size_t or similar
            index_expr, conversions = do_implicit_conversion(
                index, IntType(), "array index access"
            )
            self._indices.append(index_expr)
            self._conversion_exprs.extend(conversions)

        super().__init__(array_definition._element_type.to_unborrowed_ref())

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction
        conversion_ir = []
        for expression in self._conversion_exprs:
            conversion_ir.extend(expression.generate_ir(reg_gen))

        indices_ir = []
        for index in self._indices:
            indices_ir.append(index.ir_ref_with_type_annotation)

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        self.result_reg = next(reg_gen)
        return [
            *conversion_ir,
            f"%{self.result_reg} = getelementptr inbounds {self._array_type.ir_type},"
            f" {self._array_ptr.ir_ref_with_type_annotation}, {', '.join(indices_ir)}",
        ]

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        indices = ", ".join(map(repr, self._indices))
        return f"ArrayIndexAccess({self._array_type}[{indices}])"

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the members are initialized?
        pass

    def assert_can_write_to(self) -> None:
        pass

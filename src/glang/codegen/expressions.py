from abc import abstractmethod
from typing import Iterator, Optional

from .builtin_types import (
    BoolType,
    FunctionSignature,
    HeapArrayDefinition,
    IntType,
    SizeType,
    StackArrayDefinition,
    StructDefinition,
    format_array_dims_for_ir,
)
from .interfaces import Generatable, Type, TypedExpression, Variable
from .type_conversions import (
    assert_is_implicitly_convertible,
    do_implicit_conversion,
    get_implicit_conversion_cost,
)
from .user_facing_errors import (
    ArrayIndexCount,
    BorrowTypeError,
    CannotAssignToInitializerList,
    DoubleBorrowError,
    InvalidInitializerListConversion,
    InvalidInitializerListDeduction,
    InvalidInitializerListLength,
    OperandError,
    TypeCheckerError,
)


class ConstantExpression(TypedExpression):
    def __init__(self, cst_type: Type, value: str) -> None:
        super().__init__(cst_type, False)

        self.value = cst_type.definition.graphene_literal_to_ir_constant(value)

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
            variable.type.is_reference,
        )

        self.variable = variable
        self._ir_ref: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"VariableReference({self.variable.user_facing_name}: {self.variable.type})"
        )

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self._ir_ref
        return self._ir_ref

    def assert_can_read_from(self) -> None:
        self.variable.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self.variable.assert_can_write_to()

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        self._ir_ref = self.variable.ir_ref_without_type_annotation

        ir = []
        if self.variable.type.is_reference:
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
        if signature.return_type.is_reference:
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
        if not self.signature.return_type.definition.is_void:
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
    def __init__(self, expr: TypedExpression, is_explicitly_constant: bool) -> None:
        self._expr = expr

        if expr.underlying_type.is_reference:
            raise DoubleBorrowError(expr.underlying_type.format_for_output_to_user())

        if not expr.is_indirect_pointer_to_type:
            raise BorrowTypeError(expr.underlying_type.format_for_output_to_user())

        # TODO: test if the expression is constant, if it is fall back to a constant borrow
        self._is_constant = is_explicitly_constant

        # TODO: this is a bit more permissive that I'd like, but we (need)? to support
        #       indirect initialization by first assigning to a reference
        if not self._is_constant:
            expr.assert_can_write_to()

        super().__init__(expr.underlying_type.convert_to_reference_type(), False)

    def __repr__(self) -> str:
        return f"BorrowExpression({repr(self._expr)})"

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self._expr.ir_ref_without_type_annotation

    def assert_can_read_from(self) -> None:
        self._expr.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        # TODO: can this assert be reached by a user?
        assert not self._is_constant
        self._expr.assert_can_write_to()


class StructMemberAccess(TypedExpression):
    def __init__(self, lhs: TypedExpression, member_name: str) -> None:
        self._member_name = member_name
        self._lhs = lhs

        self._struct_type = lhs.underlying_type.convert_to_value_type()
        underlying_definition = lhs.underlying_type.definition
        if not isinstance(underlying_definition, StructDefinition):
            raise TypeCheckerError(
                "struct member access",
                lhs.underlying_type.format_for_output_to_user(),
                "{...}",
            )

        self._index, self._member_type = underlying_definition.get_member_by_name(
            member_name
        )

        # If the member is a reference we can always calculate an address
        if self._member_type.is_reference:
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

        # Prevent double indirection, dereference the element pointer to get the
        # underlying reference
        if self._member_type.is_reference:
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
            f"StructMemberAccess({self.underlying_type.format_for_output_to_user()}"
            f".{self._member_name}: {self.underlying_type})"
        )

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the members are initialized?
        self._lhs.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        if not self.has_address:
            raise OperandError("cannot modify temporary struct")

        self._lhs.assert_can_write_to()
        # TODO: check if the reference is const


class ArrayIndexAccess(TypedExpression):
    def __init__(
        self, array_ptr: TypedExpression, indices: list[TypedExpression]
    ) -> None:
        # NOTE: needs address since getelementptr must be used for runtime indexing
        assert array_ptr.has_address

        self._type_of_array: Type = array_ptr.underlying_type.convert_to_value_type()
        self._array_ptr = array_ptr

        array_definition = self._type_of_array.definition
        if not isinstance(
            array_definition, (StackArrayDefinition, HeapArrayDefinition)
        ):
            raise TypeCheckerError(
                "array index access",
                array_ptr.underlying_type.format_for_output_to_user(),
                "T[...]",
            )

        if isinstance(array_definition, StackArrayDefinition):
            if len(array_definition.dimensions) != len(indices):
                raise ArrayIndexCount(
                    self._type_of_array.format_for_output_to_user(),
                    len(indices),
                    len(array_definition.dimensions),
                )
        else:
            if len(indices) != 1 + len(array_definition.known_dimensions):
                raise ArrayIndexCount(
                    self._type_of_array.format_for_output_to_user(), len(indices), 1
                )

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

        super().__init__(self._element_type.convert_to_value_type(), True)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction
        array_def = self._type_of_array.definition
        assert isinstance(array_def, (StackArrayDefinition, HeapArrayDefinition))

        conversion_ir = self.expand_ir(self._conversion_exprs, reg_gen)

        # To access a stack array, we need to dereference the pointer returend
        # by `alloca` to get the address of the first element, and then we can
        # index into it. For heap arrays, we already have its address so we can
        # index it immediately.
        if isinstance(array_def, StackArrayDefinition):
            indices_ir = [
                ConstantExpression(SizeType(), "0").ir_ref_with_type_annotation
            ]
            gep_type_ir = self._type_of_array.ir_type
        else:
            indices_ir = []
            gep_type_ir = format_array_dims_for_ir(
                array_def.known_dimensions, array_def.member
            )

        for index in self._indices:
            indices_ir.append(index.ir_ref_with_type_annotation)

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        self.result_reg = next(reg_gen)
        ir = [
            *conversion_ir,
            f"%{self.result_reg} = getelementptr inbounds {gep_type_ir},"
            f" {self._array_ptr.ir_ref_with_type_annotation}, {', '.join(indices_ir)}",
        ]

        if self._element_type.is_reference:
            self.result_reg = self.dereference_double_indirection(reg_gen, ir)

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        indices = ", ".join(map(repr, self._indices))
        return f"ArrayIndexAccess({self._array_ptr}[{indices}])"

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the individual members are initialized?
        self._array_ptr.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._array_ptr.assert_can_write_to()


class ArrayInitializer(TypedExpression):
    def __init__(self, array_type: Type, element_exprs: list[TypedExpression]) -> None:
        assert not array_type.is_reference
        assert isinstance(array_type.definition, StackArrayDefinition)
        # TODO support for multidimensional arrays?
        assert len(array_type.definition.dimensions) == 1
        assert len(element_exprs) == array_type.definition.dimensions[0]

        self._result_ref: Optional[str] = None
        self._elements: list[TypedExpression] = []
        self._conversion_exprs: list[TypedExpression] = []
        self.implicit_conversion_cost = 0
        self.result_ref: Optional[str] = None

        target_type = array_type.definition.member
        for member_expr in element_exprs:
            member, extra_exprs = do_implicit_conversion(
                member_expr, target_type, "array initialization"
            )

            conversion_cost = get_implicit_conversion_cost(member_expr, target_type)
            assert conversion_cost is not None

            self._elements.append(member)
            self._conversion_exprs.extend(extra_exprs)
            self.implicit_conversion_cost += conversion_cost

        super().__init__(array_type, False, False)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        ir: list[str] = self.expand_ir(self._conversion_exprs, reg_gen)

        previous_ref = "undef"
        for index, value in enumerate(self._elements):
            current_ref = f"%{next(reg_gen)}"
            ir.append(
                f"{current_ref} = insertvalue {self.underlying_type.ir_type} {previous_ref}, "
                f"{value.ir_ref_with_type_annotation}, {index}"
            )

            previous_ref = current_ref

        self.result_ref = previous_ref
        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_ref
        return self.result_ref

    def __repr__(self) -> str:
        return f"ArrayInitializer({self.underlying_type}, {self._elements})"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot modify temporary array")


class StructInitializer(TypedExpression):
    def __init__(self, struct_type: Type, member_exprs: list[TypedExpression]) -> None:
        assert not struct_type.is_reference
        assert isinstance(struct_type.definition, StructDefinition)
        assert len(member_exprs) == len(struct_type.definition.members)

        self._result_ref: Optional[str] = None
        self._members: list[TypedExpression] = []
        self._conversion_exprs: list[TypedExpression] = []
        self.implicit_conversion_cost = 0
        self.result_ref: Optional[str] = None

        for (_, target_type), member_expr in zip(
            struct_type.definition.members, member_exprs, strict=True
        ):
            member, extra_exprs = do_implicit_conversion(member_expr, target_type)

            conversion_cost = get_implicit_conversion_cost(member_expr, target_type)
            assert conversion_cost is not None

            self._members.append(member)
            self._conversion_exprs.extend(extra_exprs)
            self.implicit_conversion_cost += conversion_cost

        super().__init__(struct_type, False, False)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        ir: list[str] = self.expand_ir(self._conversion_exprs, reg_gen)

        previous_ref = "undef"
        for index, value in enumerate(self._members):
            current_ref = f"%{next(reg_gen)}"
            ir.append(
                f"{current_ref} = insertvalue {self.underlying_type.ir_type} {previous_ref}, "
                f"{value.ir_ref_with_type_annotation}, {index}"
            )

            previous_ref = current_ref

        self.result_ref = previous_ref
        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_ref
        return self.result_ref

    def __repr__(self) -> str:
        return f"StructInitializer({self.underlying_type}, {self._members})"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot modify temporary struct")


class InitializerList(TypedExpression):
    @abstractmethod
    def format_for_output_to_user(self) -> str:
        pass

    def get_equivalent_pure_type(self) -> Type:
        assert False

    @property
    def underlying_type(self) -> Type:
        raise InvalidInitializerListDeduction(self.format_for_output_to_user())

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert False

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        raise CannotAssignToInitializerList()

    @abstractmethod
    def get_ordered_members(self, other: Type) -> list[TypedExpression]:
        pass

    def try_convert_to_type(self, other: Type) -> tuple[int, list[TypedExpression]]:
        if not isinstance(other.definition, StructDefinition):
            raise InvalidInitializerListConversion(
                other.format_for_output_to_user(True), self.format_for_output_to_user()
            )

        ordered_members = self.get_ordered_members(other)
        struct_initializer = StructInitializer(other, ordered_members)
        return struct_initializer.implicit_conversion_cost, [struct_initializer]


class NamedInitializerList(InitializerList):
    def __init__(self, members: list[TypedExpression], names: list[str]) -> None:
        super().__init__(None, False, False)

        self._members = dict(zip(names, members, strict=True))

    def format_for_output_to_user(self) -> str:
        members = [
            f"{name}: {type_name.underlying_type.format_for_output_to_user()}"
            for name, type_name in self._members.items()
        ]
        return "{" + ", ".join(members) + "}"

    def __repr__(self) -> str:
        return f"InitializerList({list(self._members.items())})"

    def get_ordered_members(self, other: Type) -> list[TypedExpression]:
        assert isinstance(other.definition, StructDefinition)

        if len(other.definition.members) != len(self._members):
            raise InvalidInitializerListLength(
                len(self._members), len(other.definition.members), "a struct"
            )

        ordered_members: list[TypedExpression] = []
        for member_name, _ in other.definition.members:
            if member_name not in self._members:
                raise InvalidInitializerListConversion(
                    other.format_for_output_to_user(True),
                    self.format_for_output_to_user(),
                )
            ordered_members.append(self._members[member_name])

        return ordered_members


class UnnamedInitializerList(InitializerList):
    def __init__(self, members: list[TypedExpression]) -> None:
        super().__init__(None, False, False)

        self._members = members

    def format_for_output_to_user(self) -> str:
        type_names = [
            member.underlying_type.format_for_output_to_user()
            for member in self._members
        ]
        return "{" + ", ".join(type_names) + "}"

    def __repr__(self) -> str:
        return f"InitializerList({self._members})"

    def get_ordered_members(self, other: Type) -> list[TypedExpression]:
        assert isinstance(other.definition, (StructDefinition, StackArrayDefinition))

        if isinstance(other.definition, StructDefinition):
            if len(other.definition.members) != len(self._members):
                raise InvalidInitializerListLength(
                    len(self._members), len(other.definition.members), "a struct"
                )
        else:
            assert isinstance(other.definition, StackArrayDefinition)
            if len(other.definition.dimensions) != 1:
                raise InvalidInitializerListConversion(
                    other.format_for_output_to_user(True),
                    self.format_for_output_to_user(),
                )
            if (dimension := other.definition.dimensions[0]) != len(self._members):
                raise InvalidInitializerListLength(
                    len(self._members), dimension, "an array"
                )

        return self._members

    def try_convert_to_type(self, other: Type) -> tuple[int, list[TypedExpression]]:
        # Unnamed initializer lists can also be converted to stack arrays.
        if isinstance(other.definition, StackArrayDefinition):
            ordered_members = self.get_ordered_members(other)
            array_initializer = ArrayInitializer(other, ordered_members)
            return array_initializer.implicit_conversion_cost, [array_initializer]

        # If other is not a stack array, then delegate to the parent class. It
        # can handle all other cases.
        return super().try_convert_to_type(other)


class LogicalOperator(TypedExpression):
    def __init__(
        self,
        operator: str,
        label_id: int,
        lhs_expression: TypedExpression,
        rhs_generatables: list[Generatable],
        rhs_expression: TypedExpression,
    ) -> None:
        super().__init__(BoolType(), False)

        assert operator in ("and", "or")

        # FIXME these errors are not great.
        lhs_expression.assert_can_read_from()
        assert_is_implicitly_convertible(
            lhs_expression, BoolType(), f"logical {operator}"
        )

        rhs_expression.assert_can_read_from()
        assert_is_implicitly_convertible(
            rhs_expression, BoolType(), f"logical {operator}"
        )

        self.result_reg: Optional[int] = None

        self.operator = operator
        self.label_id = label_id
        self.lhs_expression = lhs_expression
        self.rhs_generatables = rhs_generatables
        self.rhs_expression = rhs_expression

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#br-instruction
        # https://llvm.org/docs/LangRef.html#phi-instruction
        ir: list[str] = []

        lhs_label = f"l{self.operator}{self.label_id}.lhs"
        rhs_label = f"l{self.operator}{self.label_id}.rhs"
        end_label = f"l{self.operator}{self.label_id}.end"

        # Logical and evaluates the RHS condition only if the LHS evaluates to
        # true. Logical or evaluates the RHS condition only if the LHS evaluates
        # to false.
        label_if_true = rhs_label if self.operator == "and" else end_label
        label_if_false = end_label if self.operator == "and" else rhs_label

        # Start by branching to a known label name. This would have been
        # entirely unnecessary if we knew what the current label name was, but
        # we don't!
        # br label <dest>
        ir.append(f"br label %{lhs_label}")

        # lhs body.
        ir.append(f"{lhs_label}:")

        # Cast lhs to bool.
        conv_lhs, extra_lhs_exprs = do_implicit_conversion(
            self.lhs_expression, BoolType()
        )
        ir.extend(self.expand_ir(extra_lhs_exprs, reg_gen))

        # Jump to either rhs_label or end_label.
        # br i1 <cond>, label <iftrue>, label <iffalse>
        ir.append(
            f"br {conv_lhs.ir_ref_with_type_annotation}, "
            f"label %{label_if_true}, label %{label_if_false}"
        )

        # rhs body.
        ir.append(f"{rhs_label}:")
        ir.extend(self.expand_ir(self.rhs_generatables, reg_gen))
        ir.extend(self.rhs_expression.generate_ir(reg_gen))

        # Cast rhs to bool.
        conv_rhs, extra_rhs_exprs = do_implicit_conversion(
            self.rhs_expression, BoolType()
        )
        ir.extend(self.expand_ir(extra_rhs_exprs, reg_gen))

        # We always go to end_label.
        # br label <dest>
        ir.append(f"br label %{end_label}")

        # end body.
        ir.append(f"{end_label}:")

        # The IR is in SSA form, so we need a phi node to obtain the result of
        # the operator in a single register.
        self.result_reg = next(reg_gen)
        # <result> = phi [fast-math-flags] <ty> [ <val0>, <label0> ], ...
        ir.append(
            f"{self.ir_ref_without_type_annotation} = phi {self.ir_type_annotation} "
            f"[ {conv_lhs.ir_ref_without_type_annotation}, %{lhs_label} ], "
            f"[ {conv_rhs.ir_ref_without_type_annotation}, %{rhs_label} ]"
        )

        return ir

    def __repr__(self) -> str:
        return (
            f"LogicalOperator({self.operator}, {self.label_id}, "
            f"{self.lhs_expression}, {self.rhs_expression})"
        )

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.result_reg
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        # TODO user-facing error.
        assert False

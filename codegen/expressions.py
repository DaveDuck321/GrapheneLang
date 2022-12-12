from functools import cached_property
from typing import Any, Iterator

from .user_facing_errors import (
    assert_else_throw,
    throw,
    OperandError,
    TypeCheckerError,
)

from .builtin_types import (
    FunctionSignature,
    IntType,
    ReferenceType,
    StringType,
    StructDefinition,
)
from .generatable import StackVariable
from .interfaces import Type, TypedExpression, Variable


class ConstantExpression(TypedExpression):
    def __init__(self, type: Type, value: Any) -> None:
        super().__init__(type)

        self.value = type.definition.cast_constant(value)

    def __repr__(self) -> str:
        return f"ConstantExpression({self.type}, {self.value})"

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"{self.value}"

    def assert_can_read_from(self) -> None:
        # Can always read the result of a constant expression.
        pass

    def assert_can_write_to(self) -> None:
        # Can never write to a constant expression (an rvalue).
        throw(OperandError(f"Cannot modify the constant {self.value}"))


class StringConstant(TypedExpression):
    def __init__(self, identifier: str) -> None:
        super().__init__(StringType())

        self.identifier = identifier

    def __repr__(self) -> str:
        return f"StringConstant({self.identifier})"

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"@{self.identifier}"

    def assert_can_read_from(self) -> None:
        # Can always read a string constant.
        pass

    def assert_can_write_to(self) -> None:
        # Can never write to a string constant.
        throw(OperandError("Cannot modify a string constant"))


class VariableReference(TypedExpression):
    def __init__(self, variable: Variable) -> None:
        super().__init__(ReferenceType(variable.type))

        self.variable = variable

    def __repr__(self) -> str:
        return f"VariableReference({self.variable.name}: {self.variable.type})"

    @cached_property
    def ir_ref_without_type(self) -> str:
        return self.variable.ir_ref_without_type

    def assert_can_read_from(self) -> None:
        # Can ready any initialized variable.
        assert isinstance(self.variable, StackVariable)
        assert_else_throw(
            self.variable.initialized,
            OperandError(f"Cannot use uninitialized variable '{self.variable.name}'"),
        )

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        assert_else_throw(
            not self.variable.constant,
            OperandError(f"Cannot modify constant variable '{self.variable.name}'"),
        )


class FunctionParameter(TypedExpression):
    def __repr__(self) -> str:
        return f"FunctionParameter({self.type})"

    def set_reg(self, reg: int) -> None:
        self.result_reg = reg

    @cached_property
    def ir_ref_without_type(self) -> str:
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

        for arg in args:
            arg.assert_can_read_from()

        self.signature = signature
        self.args = args

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#call-instruction

        self.result_reg = next(reg_gen)

        ir = f"%{self.result_reg} = call {self.signature.ir_ref}("

        args_ir = map(lambda arg: arg.ir_ref, self.args)
        ir += str.join(", ", args_ir)

        ir += ")"

        return [ir]

    def __repr__(self) -> str:
        return f"FunctionCallExpression({self.signature})"

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        # Can read any return type. Let the caller check if it's compatible.
        pass

    def assert_can_write_to(self) -> None:
        # Can write to any reference return type. TODO we don't have references
        # yet, so any attempt to write to the return value should fail for now.
        throw(OperandError(f"Cannot modify the value returned by {self.signature}"))


class StructMemberAccess(TypedExpression):
    def __init__(self, lhs: TypedExpression, member_name: str) -> None:
        self._lhs = lhs
        self._member_name = member_name

        self._struct_type = lhs.type.get_non_reference_type()

        struct_definition = self._struct_type.definition
        assert_else_throw(
            isinstance(struct_definition, StructDefinition),
            TypeCheckerError("struct member access", self._struct_type.name, "{...}"),
        )
        assert isinstance(struct_definition, StructDefinition)

        self._access_index, member_type = struct_definition.get_member(member_name)

        self._source_struct_is_reference = lhs.type.is_reference
        if self._source_struct_is_reference:
            super().__init__(ReferenceType(member_type))
        else:
            super().__init__(member_type)

    def generate_ir_for_reference_type(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#getelementptr-instruction
        index = ConstantExpression(IntType(), self._access_index)

        # <result> = getelementptr inbounds <ty>, ptr <ptrval>{, [inrange] <ty> <idx>}*
        return [
            f"%{self.result_reg} = getelementptr inbounds {self._struct_type.ir_type}, {self._lhs.ir_ref}, {index.ir_ref}",
        ]

    def generate_ir_for_value_type(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#insertvalue-instruction

        # <result> = extractvalue <aggregate type> <val>, <idx>{, <idx>}*
        return [
            f"%{self.result_reg} = extractvalue {self._lhs.ir_ref}, {self._access_index}"
        ]

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        self.result_reg = next(reg_gen)
        if self._source_struct_is_reference:
            return self.generate_ir_for_reference_type()
        else:
            return self.generate_ir_for_value_type()

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"StructMemberAccess({self._struct_type.name}.{self._member_name}: {self.type})"

    def assert_can_read_from(self) -> None:
        # TODO: can we check if the members are initialized?
        pass

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        assert_else_throw(
            not self._source_struct_is_reference,
            OperandError(f"Cannot modify temporary struct"),
        )
        if self._source_struct_is_reference:
            # TODO: check if the reference is const
            pass

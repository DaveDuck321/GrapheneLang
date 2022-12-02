from functools import cached_property
from typing import Any, Iterator

from .user_facing_errors import (
    assert_else_throw,
    OperandError,
    throw,
)

from .builtin_types import StringType, FunctionSignature
from .interfaces import Type, TypedExpression, Variable
from .generatable import StackVariable


class ConstantExpression(TypedExpression):
    def __init__(self, type: Type, value: Any) -> None:
        super().__init__(type)

        self.value = type.cast_constant(value)

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


class VariableAccess(TypedExpression):
    def __init__(self, variable: Variable) -> None:
        super().__init__(variable.type)

        self.variable = variable

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#load-instruction

        # TODO need to load other kinds of variables.
        assert isinstance(self.variable, StackVariable)

        self.result_reg = next(reg_gen)

        # Need to load this variable from the stack to a register.
        # <result> = load [volatile] <ty>, ptr <pointer>[, align <alignment>]...
        return [
            f"%{self.result_reg} = load {self.type.ir_type}, "
            f"{self.variable.ir_ref}, align {self.type.align}"
        ]

    @cached_property
    def ir_ref_without_type(self) -> str:
        assert isinstance(self.variable, StackVariable)
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"VariableAccess({self.variable.name}: {self.variable.type})"

    def assert_can_read_from(self) -> None:
        # Can ready any initialized variable.
        assert isinstance(self.variable, StackVariable)
        assert_else_throw(
            self.variable.initialized,
            OperandError(f"Cannot use uninitialized variable '{self.variable.name}'"),
        )

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        assert isinstance(self.variable, StackVariable)
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

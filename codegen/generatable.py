from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Iterator, Optional

from .user_facing_errors import (
    RedefinitionError,
    TypeCheckerError,
    assert_else_throw,
)

from .builtin_types import BoolType
from .interfaces import Generatable, TypedExpression, Variable, Type


@dataclass
class StackVariable(Variable):
    def __init__(
        self, name: str, type: Type, constant: bool, initialized: bool
    ) -> None:
        super().__init__(name, type, constant)

        self.initialized = initialized

    @cached_property
    def ir_ref(self) -> str:
        assert self.ir_reg is not None

        # alloca returns a pointer.
        return f"ptr %{self.ir_reg}"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        assert self.ir_reg is None
        self.ir_reg = next(reg_gen)

        # <result> = alloca [inalloca] <type> [, <ty> <NumElements>]
        #            [, align <alignment>] [, addrspace(<num>)]
        return [f"%{self.ir_reg} = alloca {self.type.ir_type}, align {self.type.align}"]


class Scope(Generatable):
    def __init__(self, id: int, outer_scope: Optional["Scope"] = None) -> None:
        super().__init__()

        assert id >= 0
        self._id = id

        self._outer_scope: Optional[Scope] = outer_scope
        self._variables: dict[str, StackVariable] = {}
        self._lines: list[Generatable] = []

    def add_generatable(self, line: Generatable | Iterable[Generatable]) -> None:
        if isinstance(line, Generatable):
            self._lines.append(line)
        else:
            self._lines.extend(line)

    def add_variable(self, var: StackVariable) -> None:
        # Variables can be shadowed in different (nested) scopes, but they
        # must be unique in a single scope.
        assert_else_throw(
            var.name not in self._variables, RedefinitionError("variable", var.name)
        )
        self._variables[var.name] = var

    def search_for_variable(self, var_name: str) -> Optional[StackVariable]:
        # Search this scope first.
        if var_name in self._variables:
            return self._variables[var_name]

        # Then move up the stack.
        if self._outer_scope:
            return self._outer_scope.search_for_variable(var_name)

        return None

    @cached_property
    def start_label(self) -> str:
        return f"scope_{self._id}_begin"

    @cached_property
    def end_label(self) -> str:
        return f"scope_{self._id}_end"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        contained_ir = []

        for variable in self._variables.values():
            contained_ir.extend(variable.generate_ir(reg_gen))

        for lines in self._lines:
            contained_ir.extend(lines.generate_ir(reg_gen))

        # TODO: generate the 'start' and 'end' labels when required
        #       We need to ensure each basic block has a terminating instruction
        return contained_ir

    def __repr__(self) -> str:
        return f"{{{','.join(map(repr, self._lines))}}}"


class IfStatement(Generatable):
    def __init__(self, condition: TypedExpression, scope: Scope) -> None:
        super().__init__()

        condition.assert_can_read_from()

        self.condition = condition
        self.scope = scope

        assert_else_throw(
            self.condition.type.is_implicitly_convertible_to(BoolType()),
            TypeCheckerError("if condition", self.condition.type.name, BoolType().name),
        )

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#br-instruction

        # TODO: should the if statement also generate condition code?
        #       atm its added to the parent scope by parser.generate_if_statement
        # TODO: it also seems kind of strange that we generate the scope here
        # br i1 <cond>, label <iftrue>, label <iffalse>
        return [
            f"br {self.condition.ir_ref}, label %{self.scope.start_label}, label %{self.scope.end_label}",
            f"{self.scope.start_label}:",
            *self.scope.generate_ir(reg_gen),
            f"br label %{self.scope.end_label}",  # TODO: support `else` jump
            f"{self.scope.end_label}:",
        ]

    def __repr__(self) -> str:
        return f"IfStatement({self.condition} {self.scope})"


class ReturnStatement(Generatable):
    def __init__(self, returned_expr: Optional[TypedExpression] = None) -> None:
        super().__init__()

        if returned_expr is not None:
            returned_expr.assert_can_read_from()

        self.returned_expr = returned_expr

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#i-ret

        if self.returned_expr is None:
            # ret void; Return from void function
            return ["ret void"]

        # ret <type> <value>; Return a value from a non-void function
        return [f"ret {self.returned_expr.ir_ref}"]

    def __repr__(self) -> str:
        return f"ReturnStatement({self.returned_expr})"


class VariableAssignment(Generatable):
    def __init__(self, variable: StackVariable, value: TypedExpression) -> None:
        super().__init__()

        value.assert_can_read_from()

        assert_else_throw(
            variable.type.is_implicitly_convertible_to(value.type),
            TypeCheckerError(
                "variable assignment", value.type.name, variable.type.name
            ),
        )
        self.variable = variable
        self.value = value

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#store-instruction

        # store [volatile] <ty> <value>, ptr <pointer>[, align <alignment>]...
        return [
            f"store {self.value.ir_ref}, {self.variable.ir_ref}, "
            f"align {self.variable.type.align}"
        ]

    def __repr__(self) -> str:
        return f"VariableAssignment({self.variable.name}: {self.variable.type})"

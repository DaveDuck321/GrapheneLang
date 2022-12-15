from functools import cached_property
from typing import Iterable, Iterator, Optional

from .builtin_types import BoolType
from .interfaces import Generatable, Type, TypedExpression, Variable
from .type_conversions import assert_is_implicitly_convertible, do_implicit_conversion
from .user_facing_errors import RedefinitionError, assert_else_throw


class StackVariable(Variable):
    def __init__(
        self, name: str, var_type: Type, constant: bool, initialized: bool
    ) -> None:
        super().__init__(name, var_type, constant)

        self.initialized = initialized

    @cached_property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.ir_reg is not None

        return f"%{self.ir_reg}"

    @cached_property
    def ir_ref(self) -> str:
        # alloca returns a pointer.
        return f"ptr {self.ir_ref_without_type_annotation}"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        assert self.ir_reg is None
        self.ir_reg = next(reg_gen)

        # <result> = alloca [inalloca] <type> [, <ty> <NumElements>]
        #            [, align <alignment>] [, addrspace(<num>)]
        return [
            f"%{self.ir_reg} = alloca {self.type.ir_type}, align {self.type.alignment}"
        ]


class Scope(Generatable):
    def __init__(self, scope_id: int, outer_scope: Optional["Scope"] = None) -> None:
        super().__init__()

        assert scope_id >= 0
        self._id = scope_id

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
            var.user_facing_name not in self._variables,
            RedefinitionError("variable", var.user_facing_name),
        )
        self._variables[var.user_facing_name] = var

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
        assert_is_implicitly_convertible(condition, BoolType(), "if condition")

        self.condition = condition
        self.scope = scope

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#br-instruction

        conv_condition, extra_exprs = do_implicit_conversion(self.condition, BoolType())

        ir_lines = self.expand_ir(extra_exprs, reg_gen)

        # TODO: should the if statement also generate condition code?
        #       atm its added to the parent scope by parser.generate_if_statement
        # TODO: it also seems kind of strange that we generate the scope here
        # br i1 <cond>, label <iftrue>, label <iffalse>
        ir_lines += [
            (
                f"br {conv_condition.ir_ref_with_type_annotation}, label %{self.scope.start_label},"
                f" label %{self.scope.end_label}"
            ),
            f"{self.scope.start_label}:",
            *self.scope.generate_ir(reg_gen),
            f"br label %{self.scope.end_label}",  # TODO: support `else` jump
            f"{self.scope.end_label}:",
        ]

        return ir_lines

    def __repr__(self) -> str:
        return f"IfStatement({self.condition} {self.scope})"


class ReturnStatement(Generatable):
    def __init__(
        self, return_type: Type, returned_expr: Optional[TypedExpression] = None
    ) -> None:
        super().__init__()

        if returned_expr is not None:
            returned_expr.assert_can_read_from()
            assert_is_implicitly_convertible(
                returned_expr, return_type, "return statement"
            )

        self.return_type = return_type
        self.returned_expr = returned_expr

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#i-ret

        if self.returned_expr is None:
            # ret void; Return from void function
            return ["ret void"]

        conv_returned_expr, extra_exprs = do_implicit_conversion(
            self.returned_expr, self.return_type
        )

        ir_lines = self.expand_ir(extra_exprs, reg_gen)

        # ret <type> <value>; Return a value from a non-void function
        ir_lines.append(f"ret {conv_returned_expr.ir_ref_with_type_annotation}")

        return ir_lines

    def __repr__(self) -> str:
        return f"ReturnStatement({self.returned_expr})"


class VariableAssignment(Generatable):
    def __init__(self, variable: StackVariable, value: TypedExpression) -> None:
        super().__init__()

        value.assert_can_read_from()
        assert_is_implicitly_convertible(value, variable.type, "variable assignment")

        self.variable = variable
        self.value = value

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#store-instruction

        conv_value, extra_exprs = do_implicit_conversion(self.value, self.variable.type)

        ir_lines = self.expand_ir(extra_exprs, reg_gen)

        # store [volatile] <ty> <value>, ptr <pointer>[, align <alignment>]...
        ir_lines += [
            f"store {conv_value.ir_ref_with_type_annotation}, {self.variable.ir_ref}, "
            f"align {conv_value.type.alignment}"
        ]

        return ir_lines

    def __repr__(self) -> str:
        return (
            f"VariableAssignment({self.variable.user_facing_name}:"
            f" {self.variable.type})"
        )

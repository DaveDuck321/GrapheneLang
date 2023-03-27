from functools import cached_property
from typing import Iterable, Iterator, Optional

from .builtin_types import BoolType
from .interfaces import Generatable, Type, TypedExpression, Variable
from .type_conversions import (
    assert_is_implicitly_convertible,
    do_implicit_conversion,
)
from .user_facing_errors import AssignmentToNonPointerError, RedefinitionError


class StackVariable(Variable):
    def __init__(
        self, name: str, var_type: Type, constant: bool, initialized: bool
    ) -> None:
        super().__init__(name, var_type, constant)

        self.initialized = initialized

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.ir_reg is not None

        return f"%{self.ir_reg}"

    @property
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
        if var.user_facing_name in self._variables:
            raise RedefinitionError("variable", var.user_facing_name)

        self._variables[var.user_facing_name] = var

    def search_for_variable(self, var_name: str) -> Optional[StackVariable]:
        # Search this scope first.
        if var_name in self._variables:
            return self._variables[var_name]

        # Then move up the stack.
        if self._outer_scope:
            return self._outer_scope.search_for_variable(var_name)

        return None

    @property
    def start_label(self) -> str:
        return f"scope_{self._id}_begin"

    @property
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

    def is_empty(self) -> bool:
        return not self._lines

    def is_return_guaranteed(self) -> bool:
        for line in self._lines:
            if line.is_return_guaranteed():
                # TODO: it would be nice if we could give a dead code warning here
                return True
        return False

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

    def is_return_guaranteed(self) -> bool:
        # TODO: if else is present we can return True
        return False

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

        # An implicit basic block is created immediately after a return
        next(reg_gen)

        return ir_lines

    def is_return_guaranteed(self) -> bool:
        return True

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
            f"align {conv_value.get_equivalent_pure_type().alignment}"
        ]

        return ir_lines

    def is_return_guaranteed(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"VariableAssignment({self.variable.user_facing_name}:"
            f" {self.variable.type})"
        )


class Assignment(Generatable):
    def __init__(self, dst: TypedExpression, src: TypedExpression) -> None:
        super().__init__()

        dst.assert_can_write_to()
        src.assert_can_read_from()

        if not dst.has_address:
            raise AssignmentToNonPointerError(
                dst.underlying_type.get_user_facing_name(False)
            )

        self._dst = dst
        self._src = src

        self._target_type = dst.underlying_type.convert_to_value_type()
        assert_is_implicitly_convertible(self._src, self._target_type, "assignment")

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#store-instruction
        converted_src, src_conversions = do_implicit_conversion(
            self._src, self._target_type, "assignment"
        )

        conversion_ir: list[str] = []
        conversion_ir.extend(self.expand_ir(src_conversions, reg_gen))

        # store [volatile] <ty> <value>, ptr <pointer>[, align <alignment>]...
        return [
            *conversion_ir,
            f"store {converted_src.ir_ref_with_type_annotation}, {self._dst.ir_ref_with_type_annotation}"
            f", align {self._dst.get_equivalent_pure_type().alignment}",
        ]

    def is_return_guaranteed(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Assignment({self._dst} = {self._src})"

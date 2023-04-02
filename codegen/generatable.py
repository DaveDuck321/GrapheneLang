from typing import Iterable, Iterator, Optional

from .builtin_types import BoolType, CharArrayDefinition
from .interfaces import Generatable, Type, TypedExpression, Variable
from .type_conversions import (
    assert_is_implicitly_convertible,
    do_implicit_conversion,
)
from .user_facing_errors import (
    AssignmentToNonPointerError,
    OperandError,
    RedefinitionError,
)


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

    def assert_can_read_from(self) -> None:
        # Can ready any initialized variable.
        if not self.initialized:
            raise OperandError(
                f"cannot use uninitialized variable '{self.user_facing_name}'"
            )

    def assert_can_write_to(self) -> None:
        # Can write to any non-constant variable.
        if self.constant:
            raise OperandError(
                f"cannot modify constant variable '{self.user_facing_name}'"
            )


class StaticVariable(Variable):
    def __init__(self, var_type: Type, constant: bool, graphene_literal: str) -> None:
        assert not var_type.is_borrowed_reference
        self._graphene_literal = graphene_literal
        super().__init__(None, var_type, constant)

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.ir_reg is not None
        if isinstance(self.type.definition, CharArrayDefinition):
            return f"@.str.{self.ir_reg}"
        else:
            return f"@.{self.ir_reg}"

    @property
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} {self.ir_ref_without_type_annotation}"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        additional_prefix = "unnamed_addr constant" if self.constant else "global"

        literal = self.type.graphene_literal_to_ir_constant(self._graphene_literal)
        # @<GlobalVarName> = [Linkage] [PreemptionSpecifier] [Visibility]
        #            [DLLStorageClass] [ThreadLocal]
        #            [(unnamed_addr|local_unnamed_addr)] [AddrSpace]
        #            [ExternallyInitialized]
        #            <global | constant> <Type> [<InitializerConstant>]
        #            [, section "name"] [, partition "name"]
        #            [, comdat [($name)]] [, align <Alignment>]
        #            [, no_sanitize_address] [, no_sanitize_hwaddress]
        #            [, sanitize_address_dyninit] [, sanitize_memtag]
        #            (, !name !N)*
        self.ir_reg = next(reg_gen)
        return [
            f"{self.ir_ref_without_type_annotation} = private {additional_prefix} "
            f"{self.type.ir_type} {literal}"
        ]

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        if self.constant:
            raise OperandError(
                f"cannot modify constant variable '{self.user_facing_name}'"
            )


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


class IfElseStatement(Generatable):
    def __init__(
        self,
        condition: TypedExpression,
        if_scope: Scope,
        else_scope: Scope,
    ) -> None:
        super().__init__()

        condition.assert_can_read_from()
        assert_is_implicitly_convertible(condition, BoolType(), "if condition")

        self.condition = condition
        self.if_scope = if_scope
        self.else_scope = else_scope

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#br-instruction

        conv_condition, extra_exprs = do_implicit_conversion(self.condition, BoolType())

        # br i1 <cond>, label <iftrue>, label <iffalse>
        return [
            *self.expand_ir(extra_exprs, reg_gen),
            (
                f"br {conv_condition.ir_ref_with_type_annotation}, label %{self.if_scope.start_label},"
                f" label %{self.else_scope.start_label}"
            ),
            # If body
            f"{self.if_scope.start_label}:",
            *self.if_scope.generate_ir(reg_gen),
            f"br label %{self.else_scope.end_label}",
            # Else body
            f"{self.else_scope.start_label}:",
            *self.else_scope.generate_ir(reg_gen),
            f"br label %{self.else_scope.end_label}",
            f"{self.else_scope.end_label}:",
        ]

    def is_return_guaranteed(self) -> bool:
        return (
            self.if_scope.is_return_guaranteed()
            and self.else_scope.is_return_guaranteed()
        )

    def __repr__(self) -> str:
        return f"IfElseStatement({self.condition} {self.if_scope} {self.else_scope})"


class WhileStatement(Generatable):
    def __init__(
        self,
        new_scope_id: int,
        condition: TypedExpression,
        condition_generatable: list[Generatable],
        inner_scope: Scope,
    ) -> None:
        super().__init__()

        condition.assert_can_read_from()
        assert_is_implicitly_convertible(condition, BoolType(), "while condition")

        self.start_label = f"while_{new_scope_id}_begin"
        self.end_label = f"while_{new_scope_id}_end"

        self.condition = condition
        self.condition_generatable = condition_generatable
        self.inner_scope = inner_scope

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#br-instruction

        conv_condition, extra_exprs = do_implicit_conversion(self.condition, BoolType())

        # br i1 <cond>, label <iftrue>, label <iffalse>
        return [
            f"br label %{self.start_label}",
            f"{self.start_label}:",
            # Evaluate condition
            *self.expand_ir(self.condition_generatable, reg_gen),
            *self.expand_ir(extra_exprs, reg_gen),
            f"br {conv_condition.ir_ref_with_type_annotation}, label %{self.inner_scope.start_label}, label %{self.end_label}",
            # Loop body
            f"{self.inner_scope.start_label}:",
            *self.inner_scope.generate_ir(reg_gen),
            f"br label %{self.start_label}",  # Loop
            f"{self.end_label}:",
        ]

    def is_return_guaranteed(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"While({self.condition} {self.inner_scope})"


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

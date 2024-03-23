from __future__ import annotations

from collections.abc import Iterable

from glang.codegen.builtin_types import BoolType, CharArrayDefinition, VoidType
from glang.codegen.debug import DIFile, DILocalVariable, DILocation, DIType
from glang.codegen.interfaces import (
    Generatable,
    IRContext,
    IROutput,
    LoopInfo,
    Type,
    TypedExpression,
    Variable,
)
from glang.codegen.type_conversions import (
    assert_is_implicitly_convertible,
    do_implicit_conversion,
)
from glang.codegen.user_facing_errors import (
    AssignmentToBorrowedReference,
    AssignmentToNonPointerError,
    CannotAssignToAConstant,
    FailedLookupError,
    OperandError,
    RedefinitionError,
    TypeCheckerError,
)
from glang.parser.lexer_parser import Meta


class StackVariable(Variable):
    def __init__(
        self,
        name: str,
        var_type: Type,
        is_mutable: bool,
        initialized: bool,
        meta: Meta,
        di_file: DIFile,
    ) -> None:
        super().__init__(name, var_type, is_mutable, meta, di_file)

        self.initialized = initialized

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.ir_reg is not None

        return f"%{self.ir_reg}"

    @property
    def ir_ref(self) -> str:
        # alloca returns a pointer.
        return f"ptr {self.ir_ref_without_type_annotation}"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        assert self.ir_reg is None
        self.ir_reg = ctx.next_reg()
        ir = IROutput()

        # <result> = alloca [inalloca] <type> [, <ty> <NumElements>]
        #            [, align <alignment>] [, addrspace(<num>)]
        ir.lines.append(
            f"%{self.ir_reg} = alloca {self.type.ir_type}, align {self.type.alignment}"
        )

        metadata = self.type.to_di_type(ctx.metadata_gen)
        ir.metadata.update(metadata)
        assert isinstance(metadata[-1], DIType)

        di_local_variable = DILocalVariable(
            ctx.next_meta(),
            self._name,
            None,  # FIXME specify arg for function arguments.
            ctx.scope,
            self._di_file,
            self._meta.start.line,
            metadata[-1],
        )
        ir.metadata.add(di_local_variable)

        di_location = DILocation(
            ctx.next_meta(), self._meta.start.line, self._meta.start.column, ctx.scope
        )
        ir.metadata.add(di_location)

        ir.lines.append(
            f"call void @llvm.dbg.declare(metadata {self.ir_ref}, metadata !{di_local_variable.id},"
            f" metadata !DIExpression()), !dbg !{di_location.id}"
        )

        return ir

    def assert_can_read_from(self) -> None:
        # Can ready any initialized variable.
        if not self.initialized:
            raise OperandError(
                f"cannot use uninitialized variable '{self.user_facing_name}'"
            )

    def assert_can_write_to(self) -> None:
        # TODO: this is kinda hacky, we aught to rename this function to
        # something like: `promise_will_write_to()``
        self.initialized = True


class StaticVariable(Variable):
    def __init__(
        self,
        name: str,
        var_type: Type,
        is_mutable: bool,
        graphene_literal: str,
        meta: Meta,
        di_file: DIFile,
    ) -> None:
        assert var_type.storage_kind == Type.Kind.VALUE
        self._graphene_literal = graphene_literal
        super().__init__(name, var_type, is_mutable, meta, di_file)

    @property
    def ir_ref_without_type_annotation(self) -> str:
        assert self.ir_reg is not None
        if isinstance(self.type.definition, CharArrayDefinition):
            return f"@.str.{self.ir_reg}"
        return f"@.{self.ir_reg}"

    @property
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} {self.ir_ref_without_type_annotation}"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        additional_prefix = "global" if self.is_mutable else "unnamed_addr constant"

        literal = self.type.definition.graphene_literal_to_ir_constant(
            self._graphene_literal
        )
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
        self.ir_reg = ctx.next_reg()
        return IROutput(
            [
                f"{self.ir_ref_without_type_annotation} = private {additional_prefix} "
                f"{self.type.ir_type} {literal}"
            ]
        )

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        pass


class Scope(Generatable):
    def __init__(
        self,
        scope_id: int,
        meta: Meta,
        outer_scope: Scope | None = None,
        is_inside_loop: bool = False,
    ) -> None:
        super().__init__(meta)

        assert scope_id >= 0
        self.id = scope_id

        self._outer_scope: Scope | None = outer_scope
        self._variables: dict[str, StackVariable] = {}
        self._lines: list[Generatable] = []
        self._generic_pack: tuple[str, int] | None = None
        self._allocations: list[StackVariable] = []
        self._is_inside_loop: bool = is_inside_loop

    def _record_allocation(self, var: StackVariable) -> None:
        self._allocations.append(var)

        if self._outer_scope is not None:
            self._outer_scope._record_allocation(var)

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

        self._record_allocation(var)
        self._variables[var.user_facing_name] = var

    def search_for_variable(self, var_name: str) -> StackVariable | None:
        # Search this scope first.
        if var_name in self._variables:
            return self._variables[var_name]

        # Then move up the stack.
        if self._outer_scope:
            return self._outer_scope.search_for_variable(var_name)

        return None

    def add_generic_pack(self, pack_name: str, pack_length: int) -> None:
        self._generic_pack = (pack_name, pack_length)

    def search_for_generic_pack(self, pack_name: str) -> list[StackVariable]:
        if self._generic_pack is None or self._generic_pack[0] != pack_name:
            if self._outer_scope is None:
                raise FailedLookupError("parameter pack", pack_name)
            return self._outer_scope.search_for_generic_pack(pack_name)

        pack_vars = [
            self.search_for_variable(f"{pack_name}{i}")
            for i in range(self._generic_pack[1])
        ]

        assert all(pack_vars)
        return pack_vars  # type: ignore

    @property
    def start_label(self) -> str:
        return f"scope_{self.id}_begin"

    @property
    def end_label(self) -> str:
        return f"scope_{self.id}_end"

    def generate_ir(self, ctx: IRContext) -> IROutput:
        contained_ir = IROutput()

        # Variables are allocated at the function scope (not in inner-scopes)
        #   This prevents a large loop causing a stack overflow
        if self._outer_scope is None:
            for variable in self._allocations:
                contained_ir.extend(variable.generate_ir(ctx))

        for line in self._lines:
            contained_ir.extend(line.generate_ir(ctx))
            if line.is_return_guaranteed():
                # Avoid generating dead instructions
                # TODO: warn about unreachable code
                break

        return contained_ir

    def is_empty(self) -> bool:
        return not self._lines

    def is_return_guaranteed(self) -> bool:
        for line in self._lines:
            if line.is_return_guaranteed():
                # TODO: it would be nice if we could give a dead code warning here
                return True
        return False

    def is_jump_guaranteed(self) -> bool:
        for line in self._lines:
            if line.is_jump_guaranteed():
                # TODO: it would be nice if we could give a dead code warning here
                return True
        return False

    def is_inside_loop(self) -> bool:
        if self._is_inside_loop:
            return True

        if self._outer_scope:
            return self._outer_scope.is_inside_loop()

        return False

    def __repr__(self) -> str:
        return f"{{{','.join(map(repr, self._lines))}}}"


class IfElseStatement(Generatable):
    def __init__(
        self,
        condition: TypedExpression,
        if_scope: Scope,
        else_scope: Scope,
        meta: Meta,
    ) -> None:
        super().__init__(meta)

        condition.assert_can_read_from()
        assert_is_implicitly_convertible(condition, BoolType(), "if condition")

        self.condition = condition
        self.if_scope = if_scope
        self.else_scope = else_scope

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#br-instruction
        need_label_after_statement = False

        conv_condition, extra_exprs = do_implicit_conversion(self.condition, BoolType())

        ir = self.expand_ir(extra_exprs, ctx)

        dbg = self.add_di_location(ctx, ir)

        # br i1 <cond>, label <iftrue>, label <iffalse>
        ir.lines.append(
            f"br {conv_condition.ir_ref_with_type_annotation}, "
            f"label %{self.if_scope.start_label}, label %{self.else_scope.start_label}, "
            f"{dbg}"
        )

        # If body
        ir.lines.append(f"{self.if_scope.start_label}:")
        ir.extend(self.if_scope.generate_ir(ctx))

        if not self.if_scope.is_jump_guaranteed():
            # Jump to the end of the if/else statement.
            ir.lines.append(f"br label %{self.else_scope.end_label}, {dbg}")
            need_label_after_statement = True

        # Else body
        ir.lines.append(f"{self.else_scope.start_label}:")
        ir.extend(self.else_scope.generate_ir(ctx))

        if not self.else_scope.is_jump_guaranteed():
            # Jump to the end of the if/else statement.
            ir.lines.append(f"br label %{self.else_scope.end_label}, {dbg}")
            need_label_after_statement = True

        if need_label_after_statement:
            # LLVM will complain if this is empty.
            ir.lines.append(f"{self.else_scope.end_label}:")

        return ir

    def is_return_guaranteed(self) -> bool:
        return (
            self.if_scope.is_return_guaranteed()
            and self.else_scope.is_return_guaranteed()
        )

    def is_jump_guaranteed(self) -> bool:
        return (
            self.if_scope.is_jump_guaranteed() and self.else_scope.is_jump_guaranteed()
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
        meta: Meta,
    ) -> None:
        super().__init__(meta)

        condition.assert_can_read_from()
        assert_is_implicitly_convertible(condition, BoolType(), "while condition")

        self.info = LoopInfo(f"while_{new_scope_id}_begin", f"while_{new_scope_id}_end")

        self.condition = condition
        self.condition_generatable = condition_generatable
        self.inner_scope = inner_scope

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#br-instruction

        conv_condition, extra_exprs = do_implicit_conversion(self.condition, BoolType())

        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)

        # br label <dest>
        ir.lines.extend(
            [
                f"br label %{self.info.start_label}, {dbg}",
                f"{self.info.start_label}:",
            ]
        )
        # Evaluate condition
        ir.extend(self.expand_ir(self.condition_generatable, ctx))
        ir.extend(self.expand_ir(extra_exprs, ctx))
        # br i1 <cond>, label <iftrue>, label <iffalse>
        ir.lines.append(
            f"br {conv_condition.ir_ref_with_type_annotation}, label "
            f"%{self.inner_scope.start_label}, label %{self.info.end_label}, {dbg}"
        )
        # Loop body
        ir.lines.append(f"{self.inner_scope.start_label}:")
        ctx.loop_stack.push(self.info)  # TODO clean up with a context manager?
        ir.extend(self.inner_scope.generate_ir(ctx))
        ctx.loop_stack.pop()

        # e.g. if the scope includes continue statements in all code paths.
        if not self.inner_scope.is_jump_guaranteed():
            ir.lines.append(f"br label %{self.info.start_label}, {dbg}")

        ir.lines.append(f"{self.info.end_label}:")

        return ir

    def is_return_guaranteed(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"While({self.condition} {self.inner_scope})"


class ReturnStatement(Generatable):
    def __init__(
        self,
        return_type: Type,
        meta: Meta,
        returned_expr: TypedExpression | None = None,
    ) -> None:
        super().__init__(meta)

        if returned_expr is not None:
            returned_expr.assert_can_read_from()
            assert_is_implicitly_convertible(
                returned_expr, return_type, "return statement"
            )
        elif return_type != VoidType():
            raise TypeCheckerError(
                "return statement",
                return_type.format_for_output_to_user(),
                VoidType().format_for_output_to_user(),
                maybe_missing_borrow=False,
            )

        self.return_type = return_type
        self.returned_expr = returned_expr

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#i-ret

        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)

        # We have to use return_type instead of returned_expr.underlying_type,
        # as returned_expr might be an initializer list, which throws when its
        # type is accesssed.
        if self.returned_expr is None or self.return_type == VoidType():
            # ret void; Return from void function
            ir.lines.append(f"ret void, {dbg}")
            return ir

        conv_returned_expr, extra_exprs = do_implicit_conversion(
            self.returned_expr, self.return_type
        )

        ir.extend(self.expand_ir(extra_exprs, ctx))

        # ret <type> <value>; Return a value from a non-void function
        ir.lines.append(f"ret {conv_returned_expr.ir_ref_with_type_annotation}, {dbg}")

        return ir

    def is_return_guaranteed(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ReturnStatement({self.returned_expr})"


class ContinueStatement(Generatable):
    def __init__(self, meta: Meta) -> None:
        super().__init__(meta)

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#br-instruction

        assert not ctx.loop_stack.empty()

        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)

        # br label <dest>
        ir.lines.append(f"br label %{ctx.loop_stack.peek().start_label}, {dbg}")

        return ir

    def is_return_guaranteed(self) -> bool:
        return False

    def is_jump_guaranteed(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "ContinueStatement()"


class BreakStatement(Generatable):
    def __init__(self, meta: Meta) -> None:
        super().__init__(meta)

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#br-instruction

        assert not ctx.loop_stack.empty()

        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)

        # br label <dest>
        ir.lines.append(f"br label %{ctx.loop_stack.peek().end_label}, {dbg}")

        return ir

    def is_return_guaranteed(self) -> bool:
        return False

    def is_jump_guaranteed(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "BreakStatement()"


class VariableInitialize(Generatable):
    def __init__(
        self, variable: StackVariable, value: TypedExpression, meta: Meta
    ) -> None:
        super().__init__(meta)

        value.assert_can_read_from()
        assert_is_implicitly_convertible(value, variable.type, "variable assignment")

        self.variable = variable
        self.value = value

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#store-instruction

        conv_value, extra_exprs = do_implicit_conversion(self.value, self.variable.type)

        ir = self.expand_ir(extra_exprs, ctx)

        dbg = self.add_di_location(ctx, ir)

        # store [volatile] <ty> <value>, ptr <pointer>[, align <alignment>]...
        ir.lines.append(
            f"store {conv_value.ir_ref_with_type_annotation}, {self.variable.ir_ref}, "
            f"align {conv_value.result_type_as_if_borrowed.alignment}, {dbg}"
        )

        return ir

    def is_return_guaranteed(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"VariableAssignment({self.variable.user_facing_name}:"
            f" {self.variable.type})"
        )


class Assignment(Generatable):
    def __init__(self, dst: TypedExpression, src: TypedExpression, meta: Meta) -> None:
        super().__init__(meta)

        if dst.result_type.storage_kind.is_reference():
            raise AssignmentToBorrowedReference(dst.format_for_output_to_user())

        dst.assert_can_write_to()
        src.assert_can_read_from()

        storage_kind = dst.result_type_as_if_borrowed.storage_kind

        if not storage_kind.is_reference():
            raise AssignmentToNonPointerError(dst.format_for_output_to_user())

        if not storage_kind.is_mutable_reference():
            raise CannotAssignToAConstant(
                dst.result_type_as_if_borrowed.format_for_output_to_user()
            )

        self._dst = dst
        self._src = src

        self._target_type = dst.result_type
        assert_is_implicitly_convertible(self._src, self._target_type, "assignment")

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#store-instruction
        converted_src, src_conversions = do_implicit_conversion(
            self._src, self._target_type, "assignment"
        )

        ir = self.expand_ir(src_conversions, ctx)

        dbg = self.add_di_location(ctx, ir)

        # store [volatile] <ty> <value>, ptr <pointer>[, align <alignment>]...
        ir.lines.append(
            f"store {converted_src.ir_ref_with_type_annotation}, "
            f"{self._dst.ir_ref_with_type_annotation}, "
            f"align {self._dst.result_type_as_if_borrowed.alignment}, {dbg}"
        )

        return ir

    def is_return_guaranteed(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Assignment({self._dst} = {self._src})"

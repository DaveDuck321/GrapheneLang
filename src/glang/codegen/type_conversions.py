from glang.codegen.builtin_types import (
    HeapArrayDefinition,
    IEEEFloatDefinition,
    IntegerDefinition,
    StackArrayDefinition,
)
from glang.codegen.interfaces import (
    IRContext,
    IROutput,
    StaticTypedExpression,
    Type,
    TypedExpression,
)
from glang.codegen.user_facing_errors import OperandError, TypeCheckerError


class RemoveIndirection(StaticTypedExpression):
    def __init__(self, ref: TypedExpression) -> None:
        super().__init__(ref.result_type, Type.Kind.VALUE, ref.meta)
        self.ref = ref

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#load-instruction
        self.result_reg = ctx.next_reg()
        return_type_ir = self.ref.result_type.ir_type

        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)

        # <result> = load [volatile] <ty>, ptr <pointer>[, align <alignment>]...
        ir.lines.append(
            f"%{self.result_reg} = load {return_type_ir}, "
            f"{self.ref.ir_ref_with_type_annotation}, "
            f"align {self.result_type_as_if_borrowed.alignment}, {dbg}"
        )

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"SquashIntoUnderlyingType({self.ref})"

    def assert_can_read_from(self) -> None:
        self.ref.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        raise OperandError("cannot modify a squashed value")


class PromoteNumeric(StaticTypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
        super().__init__(dest_type, Type.Kind.VALUE, src.meta)

        src_definition = src.result_type.definition

        assert not src.has_address
        assert dest_type.storage_kind == Type.Kind.VALUE
        assert src_definition.size < dest_type.size

        if isinstance(src_definition, IntegerDefinition):
            assert isinstance(dest_type.definition, IntegerDefinition)
            assert src_definition.is_signed == dest_type.definition.is_signed
            self.instruction = "sext" if src_definition.is_signed else "zext"
        else:
            assert isinstance(src_definition, IEEEFloatDefinition)
            assert isinstance(dest_type.definition, IEEEFloatDefinition)
            self.instruction = "fpext"

        self.src = src

    def generate_ir(self, ctx: IRContext) -> IROutput:
        # https://llvm.org/docs/LangRef.html#sext-to-instruction
        # https://llvm.org/docs/LangRef.html#zext-to-instruction
        # https://llvm.org/docs/LangRef.html#fpext-to-instruction

        self.result_reg = ctx.next_reg()
        ir = IROutput()
        dbg = self.add_di_location(ctx, ir)

        # <result> = {s,z,fp}ext <ty> <value> to <ty2> ; yields ty2
        ir.lines.append(
            f"%{self.result_reg} = {self.instruction} "
            f"{self.src.ir_ref_with_type_annotation} to {self.underlying_type.ir_type}, "
            f"{dbg}"
        )

        return ir

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return f"%{self.result_reg}"

    def __repr__(self) -> str:
        return f"PromoteNumeric({self.src} to {self.underlying_type})"

    def assert_can_read_from(self) -> None:
        self.src.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        # TODO this isn't very helpful.
        raise OperandError("cannot modify promoted integers")


class Reinterpret(StaticTypedExpression):
    def __init__(self, src: TypedExpression, dest_type: Type) -> None:
        assert src.has_address and not src.underlying_indirection_kind.is_reference()
        assert dest_type.storage_kind.is_reference()

        super().__init__(dest_type, Type.Kind.VALUE, src.meta)

        self._src = src

    @property
    def ir_ref_without_type_annotation(self) -> str:
        return self._src.ir_ref_without_type_annotation

    def __repr__(self) -> str:
        return f"Reinterpret({self._src} to {self.underlying_type})"

    def assert_can_read_from(self) -> None:
        self._src.assert_can_read_from()

    def assert_can_write_to(self) -> None:
        self._src.assert_can_write_to()


def implicit_conversion_impl(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[int, list[TypedExpression]]:
    """Attempt to convert expression src to type dest_type.

    Only the following conversions are allowed:
    - dereference an (non-reference) variable with an address to a value
    - initializer list -> compatible struct
    - integer promotion.
    - float promotion (TODO).

    Multiple conversions may be performed (e.g. dereference and then promote).

    If conversion is not possible, then a user-facing exception is raised.

    Args:
        src (TypedExpression): expression to convert.
        dest_type (Type): desired type.
        context (str, optional): conversion context, used in error messages.
            Defaults to "".

    Returns:
        tuple[int, list[TypedExpression]]: Tuple contains the promotion cost,
            The list contains the chain of expressions required to convert the
            expression to dest_type. The first element of the list is always src.
    """
    expr_list = [src]

    def last_expr() -> TypedExpression:
        return expr_list[-1]

    def last_type() -> Type:
        return expr_list[-1].result_type

    promotion_cost: int = 0

    # Always dereference implicit addresses
    if src.underlying_indirection_kind.is_reference():
        expr_list.append(RemoveIndirection(src))

    # Initializer lists (+ anything else that depends on the TypedExpression)
    additional_cost, exprs = src.try_convert_to_type(dest_type)
    promotion_cost += additional_cost
    expr_list.extend(exprs)

    match (last_type().storage_kind, dest_type.storage_kind):
        case Type.Kind.MUTABLE_OR_CONST_REF, dest_kind if dest_kind.is_reference():
            expr_list.append(
                Reinterpret(
                    last_expr(),
                    last_type().convert_to_storage_type(dest_kind),
                )
            )
            # Preferentially select the mutable overload.
            if dest_kind == Type.Kind.CONST_REF:
                promotion_cost += 1
        case a, b if a != b:
            # We never implicitly dereference.
            maybe_missing_borrow = (
                last_type() == dest_type.convert_to_storage_type(Type.Kind.VALUE)
                and isinstance(src, StaticTypedExpression)
                and src.was_reference_type_at_any_point
            )

            raise TypeCheckerError(
                context,
                src.format_for_output_to_user(),
                dest_type.format_for_output_to_user(True),
                maybe_missing_borrow,
            )

    # Integer promotion.
    last_def = last_type().definition
    dest_def = dest_type.definition
    if (
        isinstance(last_def, IntegerDefinition)
        and isinstance(dest_def, IntegerDefinition)
        and last_def.is_signed == dest_def.is_signed
        and last_def.size < dest_def.size
    ):
        promotion_cost += dest_def.size // last_def.size
        expr_list.append(PromoteNumeric(last_expr(), dest_type))

    # Floating point promotion
    if (
        isinstance(last_def, IEEEFloatDefinition)
        and isinstance(dest_def, IEEEFloatDefinition)
        and last_def.size < dest_def.size
    ):
        promotion_cost += dest_def.size // last_def.size
        expr_list.append(PromoteNumeric(last_expr(), dest_type))

    # Array reference equivalence
    last_array_def = last_type().convert_to_value_type().definition
    dest_array_def = dest_type.convert_to_value_type().definition
    if (
        last_type().storage_kind.is_reference()
        and last_type().storage_kind == dest_type.storage_kind
        and isinstance(last_array_def, HeapArrayDefinition | StackArrayDefinition)
        and isinstance(dest_array_def, HeapArrayDefinition | StackArrayDefinition)
        and last_array_def.member == dest_array_def.member
    ):
        if (
            isinstance(last_array_def, HeapArrayDefinition)
            and isinstance(dest_array_def, HeapArrayDefinition)
            and last_array_def.known_dimensions == dest_array_def.known_dimensions
        ):
            expr_list.append(Reinterpret(last_expr(), dest_type))

        if (
            isinstance(last_array_def, StackArrayDefinition)
            and isinstance(dest_array_def, HeapArrayDefinition)
            and last_array_def.dimensions[1:] == dest_array_def.known_dimensions
        ):
            # TODO: promotion cost going from known size to smaller/ unknown size
            expr_list.append(Reinterpret(last_expr(), dest_type))

        if (
            isinstance(last_array_def, StackArrayDefinition)
            and isinstance(dest_array_def, StackArrayDefinition)
            and last_array_def.dimensions[1:] == dest_array_def.dimensions[1:]
            and last_array_def.dimensions[0] >= dest_array_def.dimensions[0]
        ):
            # TODO: promotion cost going from known size to smaller/ unknown size
            expr_list.append(Reinterpret(last_expr(), dest_type))

    if last_type() != dest_type:
        raise TypeCheckerError(
            context,
            src.format_for_output_to_user(),
            dest_type.format_for_output_to_user(),
        )

    return promotion_cost, expr_list


def do_implicit_conversion(
    src: TypedExpression, dest_type: Type, context: str = ""
) -> tuple[TypedExpression, list[TypedExpression]]:
    _, expr_list = implicit_conversion_impl(src, dest_type, context)

    return expr_list[-1], expr_list[1:]


def assert_is_implicitly_convertible(
    expr: TypedExpression, target: Type, context: str
) -> None:
    # Just discard the return value. It will throw if the conversion fails.
    # TODO maybe we could cache the result for later.
    implicit_conversion_impl(expr, target, context)


def get_implicit_conversion_cost(
    src: Type | TypedExpression, dest_type: Type
) -> int | None:
    class Wrapper(StaticTypedExpression):
        def __init__(self, expr_type: Type) -> None:
            super().__init__(expr_type, Type.Kind.VALUE, None)

        @property
        def ir_ref_without_type_annotation(self) -> str:
            raise AssertionError

        def assert_can_read_from(self) -> None:
            pass

        def assert_can_write_to(self) -> None:
            pass

        def __repr__(self) -> str:
            return self.underlying_type.format_for_output_to_user(False)

    try:
        expression = src if isinstance(src, TypedExpression) else Wrapper(src)
        cost, _ = implicit_conversion_impl(expression, dest_type)
        return cost
    except TypeCheckerError:
        return None

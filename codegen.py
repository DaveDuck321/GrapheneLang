from abc import ABC, abstractclassmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import count
from typing import Any, Iterator, Optional

from errors import (
    FailedLookupError,
    OperandError,
    OverloadResolutionError,
    RedefinitionError,
    TypeCheckerError,
    assert_else_throw,
)


class Type(ABC):
    align = 1  # Unaligned
    ir_type = ""

    def __init__(self, name, definition) -> None:
        self.name = name
        self.definition = definition

    def __repr__(self) -> str:
        return self.name

    @cached_property
    def mangled_name(self) -> str:
        return "__T_TODO_NAME_MANGLE_TYPE"

    @abstractclassmethod
    def compatible_with(self, value: Any) -> bool:
        pass

    @abstractclassmethod
    def cast_constant(self, value: int) -> bool:
        pass

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Type):
            return self.name == other.name and self.definition == other.definition

        return False


class IntType(Type):
    align = 4
    ir_type = "i32"

    def __init__(self) -> None:
        super().__init__("int", "__builtin_int")

    def compatible_with(self, value: Any) -> bool:
        # TODO check if value fits inside an i32
        return isinstance(value, int)

    def cast_constant(self, value: int) -> int:
        assert self.compatible_with(value)
        return int(value)


class BoolType(Type):
    align = 1
    ir_type = "i1"

    def __init__(self) -> None:
        super().__init__("bool", "__builtin_bool")

    def compatible_with(self, value: Any) -> bool:
        return isinstance(value, bool)

    def cast_constant(self, value: bool) -> int:
        assert self.compatible_with(value)
        return int(value)


class StringType(Type):
    align = 1
    ir_type = "ptr"

    def __init__(self) -> None:
        super().__init__("string", "__builtin_str")

    def compatible_with(self, value: Any) -> bool:
        return isinstance(value, str)

    def cast_constant(self, value: str) -> str:
        assert self.compatible_with(value)
        return str(value)


@dataclass
class Variable:
    name: str
    type: Type

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return []


@dataclass
class StackVariable(Variable):
    constant: bool
    initialized: bool
    ir_reg: Optional[str] = None

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


class Generatable(ABC):
    def __init__(self, id: int) -> None:
        self.id = id

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return []

    @abstractclassmethod
    def __repr__(self) -> str:
        pass


class TypedExpression(Generatable):
    def __init__(self, id: int, type: Type) -> None:
        super().__init__(id)

        self.type = type
        self.result_reg: Optional[int] = None

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} {self.ir_ref_without_type}"

    @abstractclassmethod
    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"{self.result_reg}"

    @abstractclassmethod
    def assert_can_read_from(self) -> None:
        pass

    @abstractclassmethod
    def assert_can_write_to(self) -> None:
        pass


class ConstantExpression(TypedExpression):
    def __init__(self, id: int, type: Type, value: Any) -> None:
        super().__init__(id, type)

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
        assert_else_throw(False, OperandError("TODO"))


class StringConstant(TypedExpression):
    def __init__(self, id: int, identifier: str) -> None:
        super().__init__(id, StringType())

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
        assert_else_throw(False, OperandError("TODO"))


class Scope(Generatable):
    def __init__(self, id: int, outer_scope: Optional["Scope"] = None) -> None:
        super().__init__(id)

        self._outer_scope: Optional[Scope] = outer_scope
        self._variables: dict[str, Variable] = {}
        self._lines: list[Generatable] = []

    def add_generatable(self, line: Generatable | Iterator[Generatable]) -> None:
        if isinstance(line, Generatable):
            self._lines.append(line)
        else:
            self._lines.extend(line)

    def add_variable(self, var: Variable) -> None:
        # Variables can be shadowed in different (nested) scopes, but they
        # must be unique in a single scope.
        assert_else_throw(
            var.name not in self._variables, RedefinitionError("variable", var.name)
        )
        self._variables[var.name] = var

    def search_for_variable(self, var_name: str) -> Optional[Variable]:
        # Search this scope first.
        if var_name in self._variables:
            return self._variables[var_name]

        # Then move up the stack.
        if self._outer_scope:
            return self._outer_scope.search_for_variable(var_name)

        return None

    def get_start_label(self) -> str:
        return f"scope_{self.id}_begin"

    def get_end_label(self) -> str:
        return f"scope_{self.id}_end"

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
    def __init__(self, id: int, condition: TypedExpression, scope: Scope) -> None:
        super().__init__(id)

        condition.assert_can_read_from()

        self.condition = condition
        self.scope = scope

        assert_else_throw(
            self.condition.type == BoolType(),
            TypeCheckerError("if condition", self.condition.type.name, BoolType().name),
        )

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#br-instruction

        # TODO: should the if statement also generate condition code?
        #       atm its added to the parent scope by parser.generate_if_statement
        # TODO: it also seems kind of strange that we generate the scope here
        # br i1 <cond>, label <iftrue>, label <iffalse>
        return [
            f"br {self.condition.ir_ref}, label %{self.scope.get_start_label()}, label %{self.scope.get_end_label()}",
            f"{self.scope.get_start_label()}:",
            *self.scope.generate_ir(reg_gen),
            f"br label %{self.scope.get_end_label()}",  # TODO: support `else` jump
            f"{self.scope.get_end_label()}:",
        ]

    def __repr__(self) -> str:
        return f"IfStatement({self.condition} {self.scope})"


class ReturnStatement(Generatable):
    def __init__(
        self, id: int, returned_expr: Optional[TypedExpression] = None
    ) -> None:
        super().__init__(id)

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
    def __init__(
        self, id: int, variable: StackVariable, value: TypedExpression
    ) -> None:
        super().__init__(id)

        value.assert_can_read_from()

        assert_else_throw(
            variable.type == value.type,
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


class VariableAccess(TypedExpression):
    def __init__(self, id: int, variable: Variable) -> None:
        super().__init__(id, variable.type)

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


@dataclass
class FunctionSignature:
    name: str
    arguments: list[Type]
    return_type: Type
    foreign: bool = False

    def is_main(self) -> bool:
        return self.name == "main"

    def is_foreign(self) -> bool:
        return self.foreign

    @cached_property
    def mangled_name(self) -> str:
        # main() is immune to name mangling (irrespective of arguments)
        if self.is_main() or self.is_foreign():
            return self.name

        arguments_mangle = [arg.mangled_name for arg in self.arguments]

        # FIXME separator
        arguments_mangle = "".join(arguments_mangle)

        # Name mangle operators into digits
        legal_name_mangle = []
        for char in self.name:
            if char.isdigit():
                legal_name_mangle.append(char)
            else:
                legal_name_mangle.append(f"__O{ord(char)}")

        return f"__{''.join(legal_name_mangle)}__ARGS__{arguments_mangle}"

    def __repr__(self) -> str:
        readable_arg_names = ", ".join(map(repr, self.arguments))
        if self.is_foreign():
            return f"foreign {self.name}: ({readable_arg_names}) -> {self.return_type.name}"
        else:
            return f"function {self.name}: ({readable_arg_names}) -> {self.return_type.name}"


class Function:
    def __init__(
        self, name: str, parameters: list[Variable], return_type: Type, is_foreign: bool
    ) -> None:
        self._signature = FunctionSignature(
            name, [var.type for var in parameters], return_type, is_foreign
        )

        self.expr_id_iter = count()
        self.top_level_scope = Scope(self.get_next_expr_id())

    def __repr__(self) -> str:
        return repr(self._signature)

    @cached_property
    def mangled_name(self) -> str:
        return self._signature.mangled_name

    def get_signature(self) -> FunctionSignature:
        return self._signature

    def is_foreign(self) -> bool:
        return self._signature.is_foreign()

    def get_next_expr_id(self) -> int:
        return next(self.expr_id_iter)

    def generate_declaration(self) -> list[str]:
        ir = f"declare dso_local {self._signature.return_type.ir_type} @{self.mangled_name}("

        args_ir = [arg.ir_type for arg in self._signature.arguments]
        ir += str.join(", ", args_ir)

        # XXX nounwind indicates that the function never raises an exception.
        ir += ") nounwind"

        return [ir]

    def generate_definition(self) -> list[str]:
        lines: list[str] = []
        reg_gen = count(1)  # First register is %1

        # FIXME generate argument list
        lines.append(f"define dso_local i32 @{self.mangled_name}() {{")

        lines.extend(self.top_level_scope.generate_ir(reg_gen))

        lines.append("}")

        return lines

    def generate_ir(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#functions
        if self.is_foreign():
            assert not self.top_level_scope._lines
            return self.generate_declaration()

        return self.generate_definition()

    @cached_property
    def ir_ref(self) -> str:
        return f"{self._signature.return_type.ir_type} @{self.mangled_name}"


class FunctionCallExpression(TypedExpression):
    def __init__(
        self, id: int, function: Function, args: list[TypedExpression]
    ) -> None:
        super().__init__(id, function.get_signature().return_type)

        for arg in args:
            arg.assert_can_read_from()

        self.function = function
        self.args = args

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#call-instruction

        fn = self.function
        self.result_reg = next(reg_gen)

        ir = f"%{self.result_reg} = call {fn.ir_ref}("

        args_ir = map(lambda arg: arg.ir_ref, self.args)
        ir += str.join(", ", args_ir)

        ir += ")"

        return [ir]

    def __repr__(self) -> str:
        return f"FunctionCallExpression({self.function})"

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        # Can read any return type. Let the caller check if it's compatible.
        pass

    def assert_can_write_to(self) -> None:
        # Can write to any reference return type. TODO we don't have references
        # yet, so any attempt to write to the return value should fail for now.
        assert_else_throw(
            False,
            OperandError(f"TODO: FunctionCallExpression.assert_can_write_to"),
        )


class AddExpression(TypedExpression):
    def __init__(self, id: int, arguments: list[TypedExpression]) -> None:
        lhs, rhs = arguments
        super().__init__(id, lhs.type)

        assert lhs.type == rhs.type
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"Add({self._lhs} + {self._rhs})"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#add-instruction
        self.result_reg = next(reg_gen)

        # <result> = add nuw nsw <ty> <op1>, <op2>  ; yields ty:result
        return [
            f"%{self.result_reg} = add nuw nsw {self._lhs.ir_ref}, {self._rhs.ir_ref_without_type}"
        ]

    @cached_property
    def ir_ref_without_type(self) -> str:
        return f"%{self.result_reg}"

    def assert_can_read_from(self) -> None:
        pass

    def assert_can_write_to(self) -> None:
        assert_else_throw(
            False, OperandError("Cannot assign to `__builtin_add(..., ...)`")
        )


class FunctionSymbolTable:
    def __init__(self) -> None:
        self.foreign_functions: list[Function] = []
        self.graphene_functions: list[Function] = []
        self._functions: dict[str, list[Function]] = defaultdict(list)

    def add_function(self, fn_to_add: Function) -> None:
        fn_to_add_signature = fn_to_add.get_signature()
        matched_functions = self._functions[fn_to_add_signature.name]

        def get_printable_sig(sig: FunctionSignature) -> str:
            return f"{sig.name}: ({', '.join(map(str, sig.arguments))})"

        if fn_to_add_signature.is_foreign() and len(matched_functions) > 0:
            RedefinitionError(
                "non-overloadable foreign function",
                get_printable_sig(fn_to_add_signature),
            )

        for target in matched_functions:
            assert_else_throw(
                target.is_foreign(),
                RedefinitionError(
                    "non-overloadable foreign function",
                    get_printable_sig(target.get_signature()),
                ),
            )
            assert_else_throw(
                target._signature.arguments != fn_to_add_signature.arguments,
                RedefinitionError("function", get_printable_sig(fn_to_add_signature)),
            )

        matched_functions.append(fn_to_add)
        if fn_to_add.is_foreign():
            self.foreign_functions.append(fn_to_add)
        else:
            self.graphene_functions.append(fn_to_add)

    def lookup_function(self, fn_name: str, fn_args: list[Type]):
        candidate_functions = self._functions[fn_name]

        readable_arg_names = ", ".join(map(repr, fn_args))
        assert_else_throw(
            len(candidate_functions) > 0,
            FailedLookupError(
                "function", f"function {fn_name}: ({readable_arg_names}) -> ..."
            ),
        )

        for function in candidate_functions:
            if function.get_signature().arguments == fn_args:
                return function

        raise OverloadResolutionError(fn_name, readable_arg_names)


BUILTIN_METHODS = {
    "__builtin_add": AddExpression,
}


class Program:
    def __init__(self) -> None:
        super().__init__()

        self._function_table = FunctionSymbolTable()
        self._types: dict[str, Type] = {}
        self._strings: dict[str, str] = {}

        self._has_main: bool = False

        # TODO: where to add these magic defaults?
        self.add_type(IntType())
        self.add_type(BoolType())
        self.add_type(StringType())

    def lookup_type(self, name: str) -> Type:
        assert_else_throw(
            name in self._types, FailedLookupError("type", f"typedef {name} : ...")
        )
        return self._types[name]

    def lookup_call_expression(
        self, id: int, fn_name: str, fn_args: list[TypedExpression]
    ) -> TypedExpression:
        if fn_name in BUILTIN_METHODS:
            return BUILTIN_METHODS[fn_name](id, fn_args)

        arg_types = [arg.type for arg in fn_args]
        function = self._function_table.lookup_function(fn_name, arg_types)
        return FunctionCallExpression(id, function, fn_args)

    def add_function(self, function: Function) -> None:
        self._function_table.add_function(function)

    def add_type(self, type: Type) -> None:
        assert_else_throw(
            type.name not in self._types, RedefinitionError("type", type.name)
        )
        self._types[type.name] = type

    @staticmethod
    def _get_string_identifier(index: int) -> str:
        assert index >= 0
        return f".str.{index}"

    def add_string(self, string: str) -> str:
        id = self._get_string_identifier(len(self._strings))
        self._strings[id] = string

        return id

    def generate_ir(self, target="x86_64-pc-linux-gnu") -> list[str]:
        lines: list[str] = []

        lines.append(f'target triple = "{target}"')

        for string_id, string in self._strings.items():
            # TODO encode string correctly
            lines.append(
                f'@{string_id} = private unnamed_addr constant [{len(string) + 1} x i8] c"{string}\\00"'
            )

        for fn in self._function_table.foreign_functions:
            lines.extend(fn.generate_ir())

        for fn in self._function_table.graphene_functions:
            lines.extend(fn.generate_ir())

        return lines

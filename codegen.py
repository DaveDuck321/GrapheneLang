from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import count
from typing import Any, Iterator, Optional


class Type(ABC):
    align = 1  # Unaligned
    ir_type = ""

    def __init__(self, name, definition) -> None:
        self.name = name
        self.definition = definition

    @classmethod
    def __repr__(cls) -> str:
        return cls.ir_type

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

    @cached_property
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


class Expression(ABC):
    def __init__(self, id: int) -> None:
        self.id = id

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return []

    @abstractclassmethod
    def __repr__(self) -> str:
        pass


class TypedExpression(Expression):
    def __init__(self, id: int, type: Type) -> None:
        super().__init__(id)

        self.type = type
        self.result_reg: Optional[int] = None

    @abstractclassmethod
    @cached_property
    def ir_ref(self) -> str:
        pass


class ConstantExpression(TypedExpression):
    def __init__(self, id: int, type: Type, value: Any) -> None:
        super().__init__(id, type)

        self.value = type.cast_constant(value)

    def __repr__(self) -> str:
        return f"ConstantExpression({self.type}, {self.value})"

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} {self.value}"


class StringConstant(TypedExpression):
    def __init__(self, id: int, identifier: str) -> None:
        super().__init__(id, StringType())

        self.identifier = identifier

    def __repr__(self) -> str:
        return f"StringConstant({self.identifier})"

    @cached_property
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} @{self.identifier}"


class Scope(Expression):
    def __init__(self, id: int, outer_scope: Optional["Scope"] = None) -> None:
        super().__init__(id)

        self._outer_scope: Optional[Scope] = outer_scope
        self._variables: dict[Variable] = {}
        self._expressions: list[Expression] = []

    def add_expression(self, expr: Expression | Iterator[Exception]) -> None:
        if isinstance(expr, Expression):
            self._expressions.append(expr)
        else:
            self._expressions.extend(expr)

    def add_variable(self, var: Variable) -> None:
        # Variables can be redeclared in different (nested) scopes, but they
        # must be unique in a single scope.
        assert var.name not in self._variables

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
        return f"SCOPE_{self.id}_BEGIN"

    def get_end_label(self) -> str:
        return f"SCOPE_{self.id}_END"

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        subexpression_ir = []

        for variable in self._variables.values():
            subexpression_ir.extend(variable.generate_ir(reg_gen))

        for expression in self._expressions:
            subexpression_ir.extend(expression.generate_ir(reg_gen))

        # TODO: generate the 'start' and 'end' labels when required
        #       We need to ensure each basic block has a terminating instruction
        return subexpression_ir

    def __repr__(self) -> str:
        return f"{{{','.join(map(repr, self._expressions))}}}"


class ReturnExpression(Expression):
    def __init__(self, id: int, returned_expr: Optional[Expression] = None) -> None:
        super().__init__(id)

        self.returned_expr = returned_expr

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#i-ret

        ret_expr = self.returned_expr

        if not ret_expr:
            # ret void; Return from void function
            return ["ret void"]

        # FIXME clean-up logic below with ir_ref.

        if isinstance(ret_expr, ConstantExpression):
            # ret <type> <value>; Return a value from a non-void function
            return [f"ret {ret_expr.type.ir_type} {ret_expr.value}"]

        assert isinstance(ret_expr, TypedExpression)

        # ret <type> <value>; Return a value from a non-void function
        return [f"ret {ret_expr.type} %{ret_expr.result_reg}"]

    def __repr__(self) -> str:
        return f"ReturnExpression({self.returned_expr})"


class VariableAssignment(Expression):
    def __init__(
        self, id: int, variable: StackVariable, value: TypedExpression
    ) -> None:
        super().__init__(id)

        assert variable.type == value.type

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
    def ir_ref(self) -> str:
        assert isinstance(self.variable, StackVariable)
        return f"{self.type.ir_type} %{self.result_reg}"

    def __repr__(self) -> str:
        return f"VariableAccess({self.variable.name}: {self.variable.type})"


class FunctionSignature:
    def __init__(
        self, name: str, arguments: list[Variable], foreign: bool = False
    ) -> None:
        self._name = name
        self._arguments = arguments
        self._foreign = foreign

    def is_main(self) -> bool:
        return self._name == "main"

    def is_foreign(self) -> bool:
        return self._foreign

    @cached_property
    def mangled_name(self) -> str:
        # main() is immune to name mangling (irrespective of arguments)
        if self.is_main() or self.is_foreign():
            return self._name

        arguments_mangle = [arg.type.name for arg in self._arguments]

        # FIXME separator
        arguments_mangle = "".join(arguments_mangle)
        return f"__{self._name}__ARGS__{arguments_mangle}"


class Function:
    def __init__(self, signature: FunctionSignature, return_type: Type) -> None:
        self._signature = signature
        self._return_type = return_type

        self.expr_id_iter = count()
        self.top_level_scope = Scope(self.get_next_expr_id())

    def __repr__(self) -> str:
        return self.mangled_name

    @cached_property
    def mangled_name(self):
        return self._signature.mangled_name

    def is_foreign(self):
        return self._signature.is_foreign()

    def get_next_expr_id(self) -> int:
        return next(self.expr_id_iter)

    def generate_declaration(self) -> list[str]:
        ir = f"declare dso_local {self._return_type.ir_type} @{self}("

        args_ir = [arg.type.ir_type for arg in self._signature._arguments]
        ir += str.join(", ", args_ir)

        # XXX nounwind indicates that the function never raises an exception.
        ir += ") nounwind"

        return [ir]

    def generate_definition(self) -> list[str]:
        lines: list[str] = []
        reg_gen = count(1)  # First register is %1

        # FIXME generate argument list
        # FIXME #0 refers to attribute group 0, which we don't generate
        lines.append(f"define dso_local i32 @{self}() #0 {{")

        lines.extend(self.top_level_scope.generate_ir(reg_gen))

        lines.append("}")

        return lines

    def generate_ir(self) -> list[str]:
        # https://llvm.org/docs/LangRef.html#functions
        if self.is_foreign():
            assert not self.top_level_scope._expressions
            return self.generate_declaration()

        return self.generate_definition()

    @cached_property
    def ir_ref(self) -> str:
        return f"{self._return_type} @{self}"


class FunctionCallExpression(TypedExpression):
    def __init__(
        self, id: int, function: Function, args: list[TypedExpression]
    ) -> None:
        super().__init__(id, function._return_type)

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
    def ir_ref(self) -> str:
        return f"{self.type.ir_type} %{self.result_reg}"


class Program:
    def __init__(self) -> None:
        super().__init__()

        self._functions: dict[str, Function] = {}
        self._foreign_functions: dict[str, Function] = {}
        self._types: dict[str, Type] = {}
        self._strings: dict[str, str] = {}

        self._has_main: bool = False

        # TODO: where to add these magic defaults?
        self.add_type(IntType())
        self.add_type(StringType())

    def lookup_type(self, name: str) -> Type:
        # TODO: how to do validation?
        assert name in self._types
        return self._types[name]

    def lookup_function(self, fn_sig: FunctionSignature) -> Function:
        # TODO validation
        if fn_sig._name in self._foreign_functions:
            return self._foreign_functions[fn_sig._name]
        else:
            return self._functions[fn_sig.mangled_name]

    def add_function(self, function: Function) -> None:
        # TODO: how to do validation?
        name = function.mangled_name

        if function.is_foreign():
            assert name not in self._foreign_functions
            self._foreign_functions[name] = function
        else:
            assert name not in self._functions
            self._functions[name] = function

    def add_type(self, type: Type) -> None:
        # TODO: how to do validation?
        assert type.name not in self._types
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

        for _, fn in self._foreign_functions.items():
            lines.extend(fn.generate_ir())

        for _, fn in self._functions.items():
            lines.extend(fn.generate_ir())

        return lines

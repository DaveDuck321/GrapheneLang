from abc import ABC, abstractclassmethod
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


class IntType(Type):
    align = 4
    ir_type = "i32"

    def __init__(self) -> None:
        super().__init__("int", "__builtin_int")

    def compatible_with(self, value: Any) -> bool:
        # TODO check if value fits inside an i32
        return isinstance(value, int)

    def cast_constant(self, value: int) -> bool:
        assert self.compatible_with(value)

        return int(value)


class Variable:
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type


class Expression(ABC):
    def __init__(self, id: int) -> None:
        self.id = id
        self.result_reg: Optional[int] = None

    @abstractclassmethod
    def generate_ir(self) -> list[str]:
        pass

    @abstractclassmethod
    def __repr__(self) -> str:
        pass


class TypedExpression(Expression):
    def __init__(self, id: int, type: Type) -> None:
        super().__init__(id)

        self.type = type


class ConstantExpression(TypedExpression):
    def __init__(self, id: int, type: Type, value: Any) -> None:
        super().__init__(id, type)

        self.value = type.cast_constant(value)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        self.result_reg = next(reg_gen)
        align = self.type.align
        ir_type = self.type.ir_type

        return [
            f"%{self.result_reg} = alloca {ir_type}, align {align}\n",
            f"store {ir_type} {self.value}, {ir_type}* %{self.result_reg}, align {align}\n",
        ]

    def __repr__(self) -> str:
        return f"ConstantExpression({self.type}, {self.value})"


class StringConstant(TypedExpression):
    def __init__(self, id: int, identifier: str) -> None:
        super().__init__(id, "ptr")

        self.identifier = identifier

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        return []

    def __repr__(self) -> str:
        return f"StringConstant({self.identifier})"


class ReturnExpression(Expression):
    def __init__(self, id: int, returned_expr: Optional[Expression]) -> None:
        super().__init__(id)

        self.returned_expr = returned_expr

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#i-ret

        ret_expr = self.returned_expr

        if not ret_expr:
            # ret void; Return from void function
            return ["ret void\n"]

        if isinstance(ret_expr, ConstantExpression):
            # ret <type> <value>; Return a value from a non-void function
            return [f"ret {ret_expr.type.ir_type} {ret_expr.value}\n"]

        assert isinstance(ret_expr, TypedExpression)

        # ret <type> <value>; Return a value from a non-void function
        return [f"ret {ret_expr.type} %{ret_expr.result_reg}\n"]

    def __repr__(self) -> str:
        return f"ReturnExpression({self.returned_expr})"


class FunctionSignature:
    def __init__(self, name: str, arguments: list[Variable]) -> None:
        self._name = name
        self._arguments = arguments

    def is_main(self) -> bool:
        return self._name == "main"

    @cached_property
    def mangled_name(self) -> str:
        # main() is immune to name mangling (irrespective of arguments)
        if self.is_main():
            return self._name

        arguments_mangle = []
        for arg in self._arguments:
            arguments_mangle.append(arg.type.name)

        arguments_mangle = "".join(arguments_mangle)
        return f"__{self._name}__ARGS__{arguments_mangle}"


class Function:
    def __init__(self, signature: FunctionSignature, return_type: Type) -> None:
        self._signature = signature
        self._return_type = return_type
        self._variables: Variable = []

        self.expressions: list[Expression] = []

    def __repr__(self) -> str:
        return self.mangled_name

    @cached_property
    def mangled_name(self):
        return self._signature.mangled_name

    def add_call_subexpression(self, name_mangle: str, argument_registers: int) -> int:
        # Returns the register of the return value
        pass

    def generate_ir(self) -> list[str]:
        lines: list[str] = []
        reg_gen = count(1)  # First register is %1

        lines.append(f"define dso_local i32 @{self}() #0 {{\n")

        for expr in self.expressions:
            lines += expr.generate_ir(reg_gen)

        lines.append("}\n")

        return lines


class FunctionCallExpression(TypedExpression):
    def __init__(self, id: int, function: Function) -> None:
        super().__init__(id, function._return_type)

        self.function = function

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#call-instruction

        fn = self.function
        self.result_reg = next(reg_gen)

        # TODO add support for arguments and function pointers
        return [f"%{self.result_reg} = call {fn._return_type.ir_type} @{fn}()\n"]

    def __repr__(self) -> str:
        return f"FunctionCallExpression({self.function})"


class Program:
    def __init__(self) -> None:
        super().__init__()

        self._functions: dict[str, Function] = {}
        self._types: dict[str, Type] = {}
        self._strings: dict[str, str] = {}

        self._has_main: bool = False

        # TODO: where to add these magic defaults?
        self.add_type(IntType())

    def lookup_type(self, name: str) -> Type:
        # TODO: how to do validation?
        assert name in self._types
        return self._types[name]

    def lookup_function(self, fn_sig: FunctionSignature) -> Function:
        name = fn_sig.mangled_name

        assert name in self._functions
        return self._functions[name]

    def add_function(self, function: Function) -> None:
        # TODO: how to do validation?
        is_main = function._signature.is_main()
        if self._has_main and is_main:
            raise RuntimeError("overloading 'main' is not allowed")
        self._has_main = is_main

        mangled_name = function.mangled_name
        assert mangled_name not in self._functions
        self._functions[mangled_name] = function

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

    def generate_ir(self) -> list[str]:
        lines: list[str] = []

        for string_id, string in self._strings.items():
            # TODO encode string correctly
            lines.append(
                f'@{string_id} = private unnamed_addr constant [{len(string) + 1} x i8] c"{string}\\00"'
            )

        for _, fn in self._functions.items():
            lines += fn.generate_ir()

        return lines

from abc import ABC, abstractclassmethod
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

    def get_mangled_name(self) -> str:
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

    @abstractclassmethod
    def generate_ir(self) -> list[str]:
        pass

    @abstractclassmethod
    def __repr__(self) -> str:
        pass


class ConstantExpression(Expression):
    def __init__(self, id: int, type: Type, value: Any) -> None:
        super().__init__(id)

        self.type = type
        self.value = type.cast_constant(value)

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        reg = next(reg_gen)
        align = self.type.align
        ir_type = self.type.ir_type

        return [
            f"%{reg} = alloca {ir_type}, align {align}\n",
            f"store {ir_type} {self.value}, {ir_type}* %{reg}, align {align}\n",
        ]

    def __repr__(self) -> str:
        return f"ConstantExpression({self.type}, {self.value})"


class ReturnExpression(Expression):
    def __init__(self, id: int, returned_expr: Optional[Expression]) -> None:
        super().__init__(id)

        self.returned_expr = returned_expr

    def generate_ir(self, reg_gen: Iterator[int]) -> list[str]:
        # https://llvm.org/docs/LangRef.html#i-ret

        # TODO generate IR for fancier return values
        if self.returned_expr:
            assert isinstance(self.returned_expr, ConstantExpression)

        # ret void; Return from void function
        if not self.returned_expr:
            return ["ret void\n"]

        # ret <type> <value>; Return a value from a non-void function
        return [f"ret {self.returned_expr.type.ir_type} {self.returned_expr.value}\n"]

    def __repr__(self) -> str:
        return f"ReturnExpression({self.returned_expr})"


class FunctionSignature:
    def __init__(self, name: str, arguments: list[Variable]) -> None:
        self._name = name
        self._arguments = arguments

    def is_main(self) -> bool:
        return self._name == "main"

    def get_mangled_name(self) -> str:
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

    def get_mangled_name(self):
        return self._signature.get_mangled_name()

    def add_call_subexpression(self, name_mangle: str, argument_registers: int) -> int:
        # Returns the register of the return value
        pass

    def generate_ir(self) -> list[str]:
        lines: list[str] = []
        reg_gen = count(1)  # First register is %1

        lines.append(f"define dso_local i32 @{self.get_mangled_name()}() #0 {{\n")

        for expr in self.expressions:
            lines += expr.generate_ir(reg_gen)

        lines.append("}\n")

        return lines


class Program:
    def __init__(self) -> None:
        super().__init__()

        self._functions: dict[str, Function] = {}
        self._types: dict[str, Type] = {}

        self._has_main: bool = False

        # TODO: where to add these magic defaults?
        self.add_type(IntType())

    def lookup_type(self, name: str) -> Type:
        # TODO: how to do validation?
        assert name in self._types
        return self._types[name]

    def add_function(self, function: Function) -> None:
        # TODO: how to do validation?
        is_main = function._signature.is_main()
        if self._has_main and is_main:
            raise RuntimeError("overloading 'main' is not allowed")
        self._has_main = is_main

        mangled_name = function.get_mangled_name()
        assert mangled_name not in self._functions
        self._functions[mangled_name] = function

    def add_type(self, type: Type) -> None:
        # TODO: how to do validation?
        assert type.name not in self._types
        self._types[type.name] = type

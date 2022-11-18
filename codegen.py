class Type:
    def __init__(self, name, definition) -> None:
        self.name = name
        self.definition = definition

    def get_mangled_name(self) -> str:
        return "__T_TODO_NAME_MANGLE_TYPE"


class IntType(Type):
    def __init__(self) -> None:
        super().__init__("int", "__builtin_int")


class Variable:
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type


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

        self.expressions = []

    def get_mangled_name(self):
        return self._signature.get_mangled_name()

    def add_call_subexpression(self, name_mangle: str, argument_registers: int) -> int:
        # Returns the register of the return value
        pass


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

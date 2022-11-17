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


class Function:
    def __init__(self, name: str, arguments: list[Variable], return_type: Type) -> None:
        super().__init__()

        self._name = name
        self._return_type = return_type
        self._arguments = arguments
        self._variables: Variable = []

        self.expressions = []

    def get_mangled_name(self):
        arguments_mangle = []
        for arg in self._arguments:
            arguments_mangle.append(arg.type.name)

        arguments_mangle = "".join(arguments_mangle)
        return f"__{self._name}__ARGS__{arguments_mangle}"

    def add_call_subexpression(self, name_mangle: str, argument_registers: int) -> int:
        # Returns the register of the return value
        pass


class Program:
    def __init__(self) -> None:
        super().__init__()

        self._functions: dict[str, Function] = {}
        self._types: dict[str, Type] = {}

        # TODO: where to add these magic defaults?
        self.add_type(IntType())

    def lookup_type(self, name: str) -> Type:
        # TODO: how to do validation?
        assert name in self._types
        return self._types[name]

    def add_function(self, function: Function) -> None:
        # TODO: how to do validation?
        mangled_name = function.get_mangled_name()
        assert mangled_name not in self._functions
        self._functions[mangled_name] = function

    def add_type(self, type: Type) -> None:
        # TODO: how to do validation?
        assert type.name not in self._types
        self._types[type.name] = type

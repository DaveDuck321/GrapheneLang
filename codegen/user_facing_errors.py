class GrapheneError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TypeCheckerError(GrapheneError):
    def __init__(self, context: str, actual: str, expected: str) -> None:
        super().__init__(
            f"Error in {context}: type '{actual}' does not match expected type '{expected}'"
        )


class RedefinitionError(GrapheneError):
    def __init__(self, thing: str, name: str) -> None:
        super().__init__(f"Error: multiple definitions of {thing} '{name}'")


class FailedLookupError(GrapheneError):
    def __init__(self, thing: str, required_definition: str) -> None:
        super().__init__(
            f"Error: could not find {thing} definition '{required_definition}'"
        )


class OverloadResolutionError(GrapheneError):
    def __init__(self, fn_name: str, arguments: str) -> None:
        # TODO: suggest available overloads here
        super().__init__(
            f"Error: overload resolution for function '{fn_name}({arguments})' failed"
        )


class OperandError(GrapheneError):
    # TODO error message.
    pass


class InvalidEscapeSequence(GrapheneError):
    def __init__(self, escaped_char: str) -> None:
        assert len(escaped_char) == 1
        super().__init__(f"Error: \\{escaped_char} is not a valid escape sequence")


class InvalidIntSize(GrapheneError):
    def __init__(
        self,
        type_name: str,
        actual_value: int,
        expected_lower: int,
        expected_upper: int,
    ) -> None:
        super().__init__(
            f"Error: {type_name} cannot store value {actual_value}, permitted range"
            f" [{expected_lower}, {expected_upper})"
        )


class GenericArgumentCountError(GrapheneError):
    def __init__(self, name: str, actual: int, expected: int) -> None:
        super().__init__(
            f"Error: generic `{name}` expects {expected} arguments but received {actual}"
        )


def throw(error: GrapheneError):
    raise error


def assert_else_throw(verify: bool, error: GrapheneError):
    if not verify:
        throw(error)

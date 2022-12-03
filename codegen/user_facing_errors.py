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
            f"Error in overload resolution for function '{fn_name}({arguments})'"
        )


class OperandError(GrapheneError):
    def __init__(self, message: str) -> None:
        # TODO error message.
        super().__init__(message)


def throw(error: GrapheneError):
    raise error


def assert_else_throw(verify: bool, error: GrapheneError):
    if not verify:
        throw(error)
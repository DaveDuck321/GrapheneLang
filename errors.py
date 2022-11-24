class GrapheneError(ValueError):
    def __init__(self, message: str, *args: list[int]) -> None:
        super().__init__(message, *args)

    def __str__(self) -> str:
        if len(self.args) > 1:
            return f"\n\n\nLine: {self.args[1]}\n\t{self.args[0]}"
        else:
            return f"\n\n{self.args[0]}"


class TypeCheckerError(GrapheneError):
    def __init__(self, context: str, actual: str, expected: str) -> None:
        super().__init__(
            f"Error in {context}: type '{actual}' does not match expected type '{expected}'"
        )


class RedefinitionError(GrapheneError):
    def __init__(self, thing: str, name: str) -> None:
        super().__init__(f"Error: multiple definitions of {thing} '{name}'")


class FailedLookupError(GrapheneError):
    def __init__(self, thing: str, name: str) -> None:
        super().__init__(f"Error: could not find definition of {thing} '{name}'")


class OverloadResolutionError(GrapheneError):
    def __init__(self, fn_name: str, arguments: str) -> None:
        # TODO: suggest available overloads here
        super().__init__(
            f"Error in overload resolution for function '{fn_name}({arguments})'"
        )


def assert_else_throw(verify: bool, error: GrapheneError):
    if not verify:
        raise error

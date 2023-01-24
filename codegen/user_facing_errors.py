from lark import Token
from typing import Optional


class GrapheneError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    @property
    def message(self) -> str:
        return str(self)


class ErrorWithLineInfo(ValueError):
    def __init__(self, message: str, line: int, context: str) -> None:
        super().__init__(message)

        self.line = line
        self.context = context
        self.file: Optional[str] = None

    @property
    def message(self) -> str:
        return str(self)


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
    def __init__(
        self, fn_name: str, arguments: str, available_overloads: list[str]
    ) -> None:
        msg = [
            f"Error: overload resolution for function call '{fn_name}({arguments})' failed"
        ]

        # TODO need a better way to indent these.
        if available_overloads:
            msg.append("   Available overloads:")

            for overload in available_overloads:
                msg.append("     - " + overload)
        else:
            msg.append("   No overloads available")

        super().__init__(str.join("\n", msg))


class AmbiguousFunctionCall(GrapheneError):
    def __init__(
        self, fn_name: str, arguments: str, candidate_1: str, candidate_2: str
    ) -> None:
        super().__init__(
            f"Error: overload resolution for function call '{fn_name}({arguments})' is ambiguous. "
            f"Equally good candidates are '{candidate_1}' and '{candidate_2}'."
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
        arguments = "argument" if expected == 1 else "arguments"

        super().__init__(
            f"Error: generic '{name}' expects {expected} {arguments} but received {actual}"
        )


class ArrayIndexCount(GrapheneError):
    def __init__(self, type_name: str, actual: int, expected: int) -> None:
        super().__init__(
            f"Error: array type '{type_name}' expects {expected} indices but received {actual}"
        )


class BorrowTypeError(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(f"Error: cannot borrow non-reference type '{type_name}'")


class AssignmentToNonPointerError(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot assign to non-reference '{type_name}' since it does not have an address"
        )


class RepeatedGenericName(ErrorWithLineInfo):
    def __init__(self, generic_name: Token, type_name: str) -> None:
        assert generic_name.line is not None

        super().__init__(
            f"Error: generic '{generic_name}' appears more than once in "
            f"the declaration of '{type_name}'",
            generic_name.line,
            f"typedef {type_name} : ...",
        )


class GenericHasGenericAnnotation(GrapheneError):
    def __init__(self, generic_name: str) -> None:
        super().__init__(
            f"Error: generic '{generic_name}' has a generic annotation "
            "(generic types may not have additional generic annotations)"
        )


class InvalidInitializerListLength(GrapheneError):
    def __init__(self, actual: int, expected: int) -> None:
        objects = "object" if actual == 1 else "objects"
        members = "member" if expected == 1 else "members"

        super().__init__(
            f"Error: initializer list with {actual} {objects} cannot be "
            f"assigned to a struct with {expected} {members}"
        )


class InvalidInitializerListAssignment(GrapheneError):
    def __init__(self, non_struct_type: str, init_list_type_name: str) -> None:
        super().__init__(
            "Error: initializer list cannot be assigned to type "
            f"'{non_struct_type}' (expected a struct type similar to '{init_list_type_name}')"
        )


class FileDoesNotExistException(GrapheneError):
    def __init__(self, file_name: str) -> None:
        super().__init__(f"Error: file '{file_name}' does not exist")


class MissingFunctionReturn(ErrorWithLineInfo):
    def __init__(self, function_name: str, line: int) -> None:
        super().__init__(
            f"Error: control flow reaches end of non-void function",
            line,
            function_name,
        )

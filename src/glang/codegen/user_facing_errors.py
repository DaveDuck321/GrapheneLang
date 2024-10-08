from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Location:
    pass


@dataclass(frozen=True)
class SourceLocation(Location):
    line: int
    file: str
    include_hierarchy: tuple[str, ...]

    def __str__(self):
        return f"File '{self.file}', line {self.line}"


@dataclass(frozen=True)
class BuiltinSourceLocation(Location):
    def __str__(self) -> str:
        return "builtin"


def format_list(items: Iterable[str]) -> str:
    # Sort for consistent error messages
    return "\n".join(f"     - {item}" for item in sorted(items))


def format_list_with_locations(items: Iterable[tuple[str, Location]]) -> str:
    longest_item = max(len(item[0]) for item in items)
    return format_list((f"{item:{longest_item+5}} ({loc})" for item, loc in items))


class GrapheneError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    @property
    def message(self) -> str:
        return str(self)


class ErrorWithLineInfo(ValueError):
    def __init__(self, message: str, line: int, context: str | None = None) -> None:
        super().__init__(message)

        self.line = line
        self.context = context

    @property
    def message(self) -> str:
        return str(self)


class ErrorWithLocationInfo(ValueError):
    def __init__(
        self, message: str, location: Location, context: str | None = None
    ) -> None:
        super().__init__(message)

        self.loc = location
        self.context = context

    @property
    def message(self) -> str:
        return str(self)


class TypeCheckerError(GrapheneError):
    def __init__(
        self,
        context: str,
        actual: str,
        expected: str,
        maybe_missing_borrow: bool = False,
    ) -> None:
        extra_context = (
            "\n    "
            f"Type '{actual}' is implicitly dereferenced here, "
            "did you mean to borrow using '&...'?"
            if maybe_missing_borrow
            else ""
        )

        super().__init__(
            f"Error in {context}: type '{actual}' does not match expected type '{expected}'"
            f"{extra_context}"
        )


class RedefinitionError(GrapheneError):
    def __init__(self, thing: str, name: str) -> None:
        super().__init__(f"Error: multiple definitions of {thing} '{name}'")


class FailedLookupError(GrapheneError):
    def __init__(self, thing: str, required_definition: str) -> None:
        super().__init__(
            f"Error: could not find {thing} definition '{required_definition}'"
        )


class SubstitutionFailure(GrapheneError):
    pass


class NonExistentTypeDefinition(SubstitutionFailure):
    def __init__(self, name_with_specialization: str) -> None:
        super().__init__(
            f"Error: no definition exists for Type '{name_with_specialization}', "
            "it may be incorrectly specialized"
        )


class PatternMatchFailed(SubstitutionFailure):
    def __init__(self, target: str, actual: str) -> None:
        super().__init__(f"Error: cannot pattern match '{actual}' to '{target}'")


class SpecializationFailed(SubstitutionFailure):
    def __init__(self, target: str, actual: str) -> None:
        super().__init__(f"Error: cannot specialize '{target}' with '{actual}'")


class PatternMatchDeductionFailure(GrapheneError):
    def __init__(self, fn_name: str, unmatched_generic: str) -> None:
        super().__init__(
            f"Error: cannot deduce generic type '{unmatched_generic}' "
            f"in '{fn_name}', manual specialization is required"
        )


class NonDeterminableSize(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot construct recursive type '{type_name}' since it "
            "has a non-determinable size"
        )


class OverloadResolutionError(GrapheneError):
    def __init__(
        self,
        fn_call: str,
        available_overloads_unordered: Iterable[str],
    ) -> None:
        # Sort for consistent error messages
        available_overloads = sorted(available_overloads_unordered)
        msg = [f"Error: overload resolution for function call '{fn_call}' failed"]

        # TODO need a better way to indent these.
        if available_overloads:
            msg.append("    Available overloads:")

            for overload in available_overloads:
                msg.append("     - " + overload)
        else:
            msg.append("    No overloads available")

        super().__init__(str.join("\n", msg))


class MultipleTypeDefinitions(GrapheneError):
    def __init__(
        self, type_format: str, candidates: Iterable[tuple[str, Location]]
    ) -> None:
        super().__init__(
            f"Error: multiple definitions of type '{type_format}':\n"
            + format_list_with_locations(candidates)
        )


class AmbiguousFunctionCall(GrapheneError):
    def __init__(
        self,
        call_format: str,
        candidates: Iterable[tuple[str, Location]],
    ) -> None:
        super().__init__(
            f"Error: function call '{call_format}' is ambiguous. "
            "Equally good candidates are:\n" + format_list_with_locations(candidates)
        )


class OperandError(GrapheneError):
    def __init__(self, message: str) -> None:
        super().__init__(f"Error: {message}")


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


class InvalidFloatLiteralTooLarge(GrapheneError):
    def __init__(
        self,
        type_name: str,
        actual_value: str,
        max_representable_with_rounding: int,
        min_unrepresentable_without_rounding: int,
    ) -> None:
        max_str = str(max_representable_with_rounding)
        min_str = str(min_unrepresentable_without_rounding)
        assert len(max_str) == len(min_str)

        first_diff = next(
            i
            for i, (char1, char2) in enumerate(zip(max_str, min_str, strict=True))
            if char1 != char2
        )

        upper_truncated = f"{max_str[0]}.{max_str[1:first_diff+1]}e+{len(max_str)-1}"

        super().__init__(
            f"Error: '{type_name}' cannot represent '{actual_value}' since it is "
            "too large and would be truncated to +infty. Only values below "
            f"{upper_truncated} can be represented."
        )


class InvalidFloatLiteralPrecision(GrapheneError):
    def __init__(
        self,
        type_name: str,
        actual_value: str,
    ) -> None:
        super().__init__(
            f"Error: '{type_name}' cannot represent '{actual_value}' since there "
            f"is insufficient precision for a normalized representation."
        )


class BorrowWithNoAddressError(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(f"Error: cannot borrow type '{type_name}' (with no address)")


class DoubleBorrowError(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot borrow '{type_name}' because it is already a reference"
        )


class DoubleReferenceError(SubstitutionFailure):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot construct reference type since '{type_name}' is already a reference"
        )


class AssignmentToNonPointerError(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot assign to non-reference '{type_name}' since it does not have an address"
        )


class AssignmentToBorrowedReference(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot assign to borrowed reference type '{type_name}'. "
            "Please do not manually borrow before an assignment."
        )


class ArrayDimensionError(GrapheneError):
    def __init__(self, array_type: str) -> None:
        super().__init__(
            f"Error: cannot construct array type '{array_type}' since it has a negative size"
        )


class RepeatedGenericName(ErrorWithLineInfo):
    def __init__(self, generic_name: str, line_number: int, type_name: str) -> None:
        super().__init__(
            f"Error: generic '{generic_name}' appears more than once in "
            f"the declaration of '{type_name}'",
            line_number,
            f"typedef {type_name} : ...",
        )


class RedeclarationWithIncorrectSpecializationCount(ErrorWithLocationInfo):
    def __init__(
        self,
        name: str,
        received: int,
        expected: int,
        declarations: Iterable[tuple[str, Location]],
        loc: Location,
    ) -> None:
        received_noun = "argument" if received == 1 else "arguments"
        expected_noun = "argument" if expected == 1 else "arguments"

        super().__init__(
            f"Error: generic type '{name}' with {received} {received_noun} "
            f"was previously declared with {expected} {expected_noun}:\n"
            + format_list_with_locations(declarations),
            loc,
        )


class IncorrectSpecializationCount(GrapheneError):
    def __init__(
        self,
        thing: Literal["type", "function"],
        name: str,
        received: int,
        expected: int,
        declarations: Iterable[tuple[str, Location]],
    ) -> None:
        received_noun = "argument" if received == 1 else "arguments"
        expected_noun = "argument" if expected == 1 else "arguments"

        super().__init__(
            f"Error: generic {thing} '{name}' received {received} {received_noun}. "
            "This conflicts with the following definitions that expect "
            f"{expected} {expected_noun}:\n" + format_list_with_locations(declarations),
        )


class GenericHasGenericAnnotation(GrapheneError):
    def __init__(self, generic_name: str) -> None:
        super().__init__(
            f"Error: generic '{generic_name}' has a generic annotation "
            "(generic types may not have additional generic annotations)"
        )


class InvalidInitializerListLength(TypeCheckerError):
    def __init__(
        self, actual: int, expected: int, target_kind: Literal["a struct", "an array"]
    ) -> None:
        objects = "object" if actual == 1 else "objects"
        members = "member" if expected == 1 else "members"

        super(GrapheneError, self).__init__(
            f"Error: initializer list with {actual} {objects} cannot be "
            f"converted to {target_kind} with {expected} {members}"
        )


class InvalidInitializerListConversion(TypeCheckerError):
    def __init__(self, struct_type: str, init_list_name: str) -> None:
        super(GrapheneError, self).__init__(
            f"Error: initializer list of the form '{init_list_name}' cannot be "
            f"converted to '{struct_type}'"
        )


class CannotAssignToInitializerList(GrapheneError):
    def __init__(self) -> None:
        super().__init__("Error: cannot assign to an initializer list")


class FileDoesNotExistException(GrapheneError):
    def __init__(self, file_name: str) -> None:
        super().__init__(f"Error: file '{file_name}' does not exist")


class FileIsAmbiguousException(GrapheneError):
    def __init__(self, relative_path: str, candidates: Iterable[str]) -> None:
        super().__init__(
            f"Error: file '{relative_path}' is ambiguous, possible candidates are:\n"
            + format_list(candidates)
        )


class CircularImportException(GrapheneError):
    def __init__(self, import_name: str, conflicting_file_name: str) -> None:
        super().__init__(
            f"Error: cannot import '{import_name}' since this would be a circular dependency: "
            f"already imported from parent '{conflicting_file_name}'"
        )


class MissingFunctionReturn(ErrorWithLineInfo):
    def __init__(self, function_name: str, line: int) -> None:
        super().__init__(
            "Error: control flow reaches end of non-void function",
            line,
            function_name,
        )


class VoidVariableDeclaration(GrapheneError):
    def __init__(
        self,
        kind: Literal["variable", "argument"],
        variable_name: str,
        full_type: str,
    ) -> None:
        super().__init__(
            f"Error: {kind} '{variable_name}' cannot have type '{full_type}'"
        )


class VoidStructDeclaration(GrapheneError):
    def __init__(
        self,
        struct_type: str,
        member_name: str,
        member_type: str,
    ) -> None:
        super().__init__(
            f"Error: struct '{struct_type}' cannot have member '{member_name}' "
            f"of type '{member_type}'"
        )


class VoidArrayDeclaration(GrapheneError):
    def __init__(
        self,
        array_type: str,
        member_type: str,
    ) -> None:
        super().__init__(
            f"Error: array '{array_type}' cannot operate on scalar '{member_type}'"
        )


class InvalidMainReturnType(GrapheneError):
    def __init__(self, actual_type: str) -> None:
        super().__init__(
            f"Error: type '{actual_type}' is not a valid return type for"
            " function 'main'; expected 'int'"
        )


class CannotAssignToAConstant(GrapheneError):
    def __init__(self, thing: str) -> None:
        super().__init__(
            "Error: left hand side of assignment operation has "
            f"non-mutable type '{thing}'"
        )


class MutableBorrowOfNonMutable(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: cannot perform mutable borrow on non-mutable type '{type_name}'"
        )


class MutableVariableContainsAReference(GrapheneError):
    def __init__(self, var_name: str, type_name: str) -> None:
        super().__init__(
            f"Error: cannot store reference type '{type_name}' in mutable variable "
            f"'{var_name}', consider using 'let {var_name} : {type_name} = ...'"
        )


class CannotUseHeapArrayInValueContext(GrapheneError):
    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"Error: heap array type '{type_name}' cannot be used as a value type."
        )


class InvalidSyntax(ErrorWithLineInfo):
    def __init__(
        self, context_lines: list[str], line_number: int, hint: str = ""
    ) -> None:
        message = "    ".join(context_lines)
        message += f"\nError: invalid syntax, {hint}"
        super().__init__(message, line_number)


class StructMemberRedeclaration(ErrorWithLineInfo):
    def __init__(self, name: str, previous_type: str, line: int) -> None:
        # TODO would be nice if we could print the struct's name (if available).
        super().__init__(
            f"Error: redeclaration of struct member `{name}` (previous "
            f"declaration `{name}: {previous_type}`)",
            line,
            "struct definition",
        )


class NotInLoopStatementError(GrapheneError):
    def __init__(self, statement: str) -> None:
        super().__init__(f"'{statement}' is not inside a loop")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from typing import Type as PyType
from typing import TypeVar

from .self_hosted_parser import parse

# Mirrors the data structures in `syntax_tree.c3`
# TODO: python bindings to automate this :-D


@dataclass
class FilePosition:
    line: int
    column: int


@dataclass
class Meta:
    start: FilePosition
    end: FilePosition


@dataclass
class ParsedNode:
    meta: Meta


@dataclass
class TopLevelFeature(ParsedNode):
    pass


@dataclass
class CompileTimeConstant(ParsedNode):
    pass


@dataclass
class NumericGenericIdentifier(CompileTimeConstant):
    value: str


@dataclass
class NumericIdentifier(CompileTimeConstant):
    value: int


@dataclass
class Type(ParsedNode):
    pass


@dataclass
class FunctionType(Type):
    pass


@dataclass
class ArrayType(Type):
    base_type: Type
    size: list[CompileTimeConstant]


@dataclass
class HeapArrayType(ArrayType):
    pass


@dataclass
class StackArrayType(ArrayType):
    pass


@dataclass
class ReferenceType(Type):
    value_type: Type


@dataclass
class NamedType(Type):
    name: str
    specialization: list[Type | CompileTimeConstant]


@dataclass
class StructType(Type):
    members: list[tuple[str, Type]]


@dataclass
class PackType(Type):
    type_: Type


@dataclass
class GenericDefinition(Type):
    name: str
    is_packed: bool


@dataclass
class NumericGenericDefinition(GenericDefinition):
    pass


@dataclass
class TypeGenericDefinition(GenericDefinition):
    pass


@dataclass
class FunctionDefinition(TopLevelFeature):
    name: str
    args: list[tuple[str, Type]]
    return_: Type


@dataclass
class GenericFunctionDefinition(FunctionDefinition):
    generic_definitions: list[GenericDefinition]
    specialization: list[Type | CompileTimeConstant]
    scope: "Scope"


@dataclass
class ImplicitFunction(GenericFunctionDefinition):
    pass


@dataclass
class GrapheneFunction(GenericFunctionDefinition):
    pass


@dataclass
class OperatorFunction(GenericFunctionDefinition):
    pass


@dataclass
class AssignmentFunction(GenericFunctionDefinition):
    pass


@dataclass
class ForeignFunction(FunctionDefinition):
    pass


@dataclass
class RequireOnce(TopLevelFeature):
    path: str


@dataclass
class Typedef(TopLevelFeature):
    generic_definitions: list[GenericDefinition]
    name: str
    specialization: list[Type | CompileTimeConstant]
    alias: Type


@dataclass
class LineOfCode(ParsedNode):
    pass


@dataclass
class Scope(LineOfCode):
    lines: list[LineOfCode]


@dataclass
class Expression(LineOfCode):
    pass


@dataclass
class If(LineOfCode):
    condition: Expression
    if_scope: Scope
    else_scope: Scope


@dataclass
class While(LineOfCode):
    condition: Expression
    scope: Scope


@dataclass
class For(LineOfCode):
    variable: str
    iterator: Expression
    scope: Scope


@dataclass
class Return(LineOfCode):
    expression: Optional[Expression]


@dataclass
class Assignment(LineOfCode):
    lhs: Expression
    operator: str
    rhs: Expression


@dataclass
class VariableDeclaration(LineOfCode):
    is_const: bool
    variable: str
    type_: Type
    expression: Optional[Expression]


@dataclass
class OperatorUse(Expression):
    name: str
    lhs: Expression
    rhs: Expression


@dataclass
class UnaryOperatorUse(Expression):
    name: str
    rhs: Expression


@dataclass
class LogicalOperatorUse(Expression):
    name: str
    lhs: Expression
    rhs: Expression


@dataclass
class Borrow(Expression):
    is_const: bool
    expression: Expression


@dataclass
class FunctionCall(Expression):
    name: str
    specialization: list[Type | CompileTimeConstant]
    args: list[Expression]


@dataclass
class UFCS_Call(Expression):
    expression: Expression
    name: str
    specialization: list[Type | CompileTimeConstant]
    args: list[Expression]


@dataclass
class PackExpansion(Expression):
    expression: Expression


@dataclass
class Constant(Expression):
    pass


@dataclass
class StringConstant(Constant):
    value: str


@dataclass
class FloatConstant(Constant):
    value: str


@dataclass
class IntConstant(Constant):
    value: int


@dataclass
class BoolConstant(Constant):
    value: bool


@dataclass
class GenericIdentifierConstant(Constant):
    value: str


@dataclass
class HexConstant(Constant):
    value: str


@dataclass
class NamedInitializerList(Expression):
    args: list[tuple[str, Expression]]


@dataclass
class UnnamedInitializerList(Expression):
    args: list[Expression]


@dataclass
class VariableAccess(Expression):
    name: str


@dataclass
class ArrayIndexAccess(Expression):
    expression: Expression
    indexes: list[Expression]


@dataclass
class StructIndexAccess(Expression):
    expression: Expression
    member: str


class Interpreter:
    T = TypeVar("T")

    def parse(self, thing: ParsedNode, ret_type: PyType[T] | None) -> T:
        for fn_type in type(thing).mro():
            if hasattr(self, fn_type.__name__):
                fn = getattr(self, fn_type.__name__)
                break
        else:
            assert False, f"{self} could not dispatch '{thing}'"

        result = fn(thing)
        if ret_type is not None:
            assert isinstance(result, ret_type)

        return result


def run_lexer_parser(path: Path) -> list[TopLevelFeature]:
    def object_hook(obj: dict[str, Any]) -> ParsedNode:
        class_ = globals()[obj["__type__"]]
        del obj["__type__"]
        return class_(**obj)

    parse_result = parse(path)
    return json.loads(parse_result, object_hook=object_hook)

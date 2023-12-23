from dataclasses import dataclass
from pathlib import Path


@dataclass
class Metadata:
    id: int


@dataclass
class MetadataFlag(Metadata):
    behaviour: int
    metadata: str
    value: int

    def __repr__(self) -> str:
        return f'!{{i32 {self.behaviour}, !"{self.metadata}", i32 {self.value}}}'


@dataclass
class DIFile(Metadata):
    file: Path

    def __repr__(self) -> str:
        # TODO checksum.
        path = self.file.resolve(strict=True)
        return f'!DIFile(filename: "{path.name}", directory: "{path.parent}")'


@dataclass
class DICompileUnit(Metadata):
    file: DIFile

    def __repr__(self) -> str:
        # TODO print glang version.
        # NOTE clang checks that the language is valid... so let's pretend we
        # are C.
        return (
            f"distinct !DICompileUnit(language: DW_LANG_C, file: !{self.file.id}, "
            'producer: "glang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, '
            "splitDebugInlining: false, nameTableKind: None)"
        )


@dataclass
class DISubroutineType(Metadata):
    # TODO implementation.

    def __repr__(self) -> str:
        return f"!DISubroutineType(types: !{{null, null}})"


@dataclass
class DISubprogram(Metadata):
    name: str
    linkage_name: str
    subroutine_type: DISubroutineType
    file: DIFile
    line: int
    unit: DICompileUnit
    is_definition: bool

    def __repr__(self) -> str:
        sp_flags = "spFlags: DISPFlagDefinition" if self.is_definition else ""
        return (
            f'distinct !DISubprogram(name: "{self.name}", linkageName: "{self.linkage_name}", '
            f"scope: !{self.file.id}, file: !{self.file.id}, line: {self.line}, "
            f"scopeLine: {self.line}, type: !{self.subroutine_type.id}, unit: !{self.unit.id}, "
            f"{sp_flags})"
        )


@dataclass
class DILocation(Metadata):
    line: int
    column: int
    scope: DISubprogram

    def __repr__(self) -> str:
        return f"!DILocation(line: {self.line}, column: {self.column}, scope: !{self.scope.id})"

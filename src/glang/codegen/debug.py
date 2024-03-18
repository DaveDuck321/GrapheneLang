from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from hashlib import file_digest
from inspect import getmembers
from pathlib import Path
from typing import Optional


class Tag(StrEnum):
    null = "DW_TAG_null"
    array_type = "DW_TAG_array_type"
    class_type = "DW_TAG_class_type"
    entry_point = "DW_TAG_entry_point"
    enumeration_type = "DW_TAG_enumeration_type"
    formal_parameter = "DW_TAG_formal_parameter"
    imported_declaration = "DW_TAG_imported_declaration"
    label = "DW_TAG_label"
    lexical_block = "DW_TAG_lexical_block"
    member = "DW_TAG_member"
    pointer_type = "DW_TAG_pointer_type"
    reference_type = "DW_TAG_reference_type"
    compile_unit = "DW_TAG_compile_unit"
    string_type = "DW_TAG_string_type"
    structure_type = "DW_TAG_structure_type"
    subroutine_type = "DW_TAG_subroutine_type"
    typedef = "DW_TAG_typedef"
    union_type = "DW_TAG_union_type"
    unspecified_parameters = "DW_TAG_unspecified_parameters"
    variant = "DW_TAG_variant"
    common_block = "DW_TAG_common_block"
    common_inclusion = "DW_TAG_common_inclusion"
    inheritance = "DW_TAG_inheritance"
    inlined_subroutine = "DW_TAG_inlined_subroutine"
    module = "DW_TAG_module"
    ptr_to_member_type = "DW_TAG_ptr_to_member_type"
    set_type = "DW_TAG_set_type"
    subrange_type = "DW_TAG_subrange_type"
    with_stmt = "DW_TAG_with_stmt"
    access_declaration = "DW_TAG_access_declaration"
    base_type = "DW_TAG_base_type"
    catch_block = "DW_TAG_catch_block"
    const_type = "DW_TAG_const_type"
    constant = "DW_TAG_constant"
    enumerator = "DW_TAG_enumerator"
    file_type = "DW_TAG_file_type"
    friend = "DW_TAG_friend"
    namelist = "DW_TAG_namelist"
    namelist_item = "DW_TAG_namelist_item"
    packed_type = "DW_TAG_packed_type"
    subprogram = "DW_TAG_subprogram"
    template_type_parameter = "DW_TAG_template_type_parameter"
    template_value_parameter = "DW_TAG_template_value_parameter"
    thrown_type = "DW_TAG_thrown_type"
    try_block = "DW_TAG_try_block"
    variant_part = "DW_TAG_variant_part"
    variable = "DW_TAG_variable"
    volatile_type = "DW_TAG_volatile_type"
    dwarf_procedure = "DW_TAG_dwarf_procedure"
    restrict_type = "DW_TAG_restrict_type"
    interface_type = "DW_TAG_interface_type"
    namespace = "DW_TAG_namespace"
    imported_module = "DW_TAG_imported_module"
    unspecified_type = "DW_TAG_unspecified_type"
    partial_unit = "DW_TAG_partial_unit"
    imported_unit = "DW_TAG_imported_unit"
    condition = "DW_TAG_condition"
    shared_type = "DW_TAG_shared_type"
    type_unit = "DW_TAG_type_unit"
    rvalue_reference_type = "DW_TAG_rvalue_reference_type"
    template_alias = "DW_TAG_template_alias"
    coarray_type = "DW_TAG_coarray_type"
    generic_subrange = "DW_TAG_generic_subrange"
    dynamic_type = "DW_TAG_dynamic_type"
    atomic_type = "DW_TAG_atomic_type"
    call_site = "DW_TAG_call_site"
    call_site_parameter = "DW_TAG_call_site_parameter"
    skeleton_unit = "DW_TAG_skeleton_unit"
    immutable_type = "DW_TAG_immutable_type"


class TypeKind(StrEnum):
    # DWARF attribute type encodings.
    address = "DW_ATE_address"
    boolean = "DW_ATE_boolean"
    complex_float = "DW_ATE_complex_float"
    float = "DW_ATE_float"
    signed = "DW_ATE_signed"
    signed_char = "DW_ATE_signed_char"
    unsigned = "DW_ATE_unsigned"
    unsigned_char = "DW_ATE_unsigned_char"
    imaginary_float = "DW_ATE_imaginary_float"
    packed_decimal = "DW_ATE_packed_decimal"
    numeric_string = "DW_ATE_numeric_string"
    edited = "DW_ATE_edited"
    signed_fixed = "DW_ATE_signed_fixed"
    unsigned_fixed = "DW_ATE_unsigned_fixed"
    decimal_float = "DW_ATE_decimal_float"
    UTF = "DW_ATE_UTF"
    UCS = "DW_ATE_UCS"
    ASCII = "DW_ATE_ASCII"


class ChecksumKind(StrEnum):
    MD5 = "CSK_MD5"
    SHA1 = "CSK_SHA1"
    SHA256 = "CSK_SHA256"


class ModuleFlagsBehavior(IntEnum):
    # Determines how conflicting metadata flags are handled by LLVM when two or
    # more modules are merged together.
    # https://llvm.org/docs/LangRef.html#module-flags-metadata
    Error = 1
    Warning = 2
    Require = 3
    Override = 4
    Append = 5
    AppendUnique = 6
    Max = 7
    Min = 8


@dataclass(repr=False, eq=False)
class Metadata:
    id: int

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Metadata) and self.id == value.id

    def __repr__(self) -> str:
        contents: list[str] = []

        for key, value in getmembers(self):
            # Hide implementation details.
            if key.startswith("_") or key == "id":
                continue

            if value is None:
                continue

            # Use type() instead of isinstance() to exclude subclasses of str
            # (notably, StrEnum).
            if type(value) == str:
                value = f'"{value}"'
            if isinstance(value, Metadata):
                value = f"!{value.id}"

            contents.append(f"{key}: {value}")

        return f"!{self.__class__.__name__}({str.join(', ', contents)})"


@dataclass(repr=False, eq=False)
class ModuleFlags(Metadata):
    behaviour: ModuleFlagsBehavior
    metadata: str
    value: int

    def __repr__(self) -> str:
        return f'!{{i32 {self.behaviour}, !"{self.metadata}", i32 {self.value}}}'


@dataclass(repr=False, eq=False)
class MetadataList(Metadata):
    children: Sequence[Metadata]

    def __repr__(self) -> str:
        return "!{" + ", ".join(f"!{c.id}" for c in self.children) + "}"


@dataclass(repr=False, eq=False)
class DISubrange(Metadata):
    count: int


@dataclass(repr=False, eq=False)
class DIScope(Metadata):
    pass


@dataclass(repr=False, eq=False)
class DIFile(DIScope):
    _file: Path

    @property
    def filename(self) -> str:
        return self._file.name

    @property
    def directory(self) -> str:
        return str(self._file.resolve(strict=True).parent)

    @property
    def checksum(self) -> str:
        with self._file.open("rb") as file:
            return file_digest(file, "sha256").hexdigest()

    @property
    def checksumkind(self) -> ChecksumKind:
        return ChecksumKind.SHA256


@dataclass(repr=False, eq=False)
class DICompileUnit(DIScope):
    file: DIFile

    def __repr__(self) -> str:
        # TODO print glang version.
        # NOTE clang checks that the language is valid... so let's pretend we
        # are C++ (gdb prefixes all the structs with `struct` if we use C).
        return (
            f"distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !{self.file.id}, "
            'producer: "glang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, '
            "splitDebugInlining: false, nameTableKind: None)"
        )


@dataclass(repr=False, eq=False)
class DILocation(Metadata):
    line: int
    column: int
    scope: DIScope


@dataclass(repr=False, eq=False)
class DIType(DIScope):
    name: Optional[str]
    _size_in_bits: int
    tag: Tag

    @property
    def size(self) -> int:
        return self._size_in_bits


@dataclass(repr=False, eq=False)
class DIBasicType(DIType):
    encoding: TypeKind


@dataclass(repr=False, eq=False)
class DICompositeType(DIType):
    baseType: Optional[DIType]
    elements: Optional[MetadataList]
    # TODO
    # scope: DIScope
    # file: DIFile
    # line: int

    def __post_init__(self) -> None:
        # Required for arrays, or clang segfaults.
        if self.tag == Tag.array_type:
            assert self.elements is not None
            assert self.baseType is not None


@dataclass(repr=False, eq=False)
class DIDerivedType(DIType):
    # scope: DIScope
    # file: DIFile
    # line: int
    baseType: DIType
    offset: Optional[int]


@dataclass(repr=False, eq=False)
class DISubroutineType(DIType):
    # TODO implementation.

    def __repr__(self) -> str:
        return "!DISubroutineType(types: !{null, null})"


@dataclass(repr=False, eq=False)
class DILocalVariable(Metadata):
    name: str
    arg: Optional[int]
    scope: DIScope
    file: DIFile
    line: int
    type: DIType


@dataclass(repr=False, eq=False)
class DISubprogram(DIScope):
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

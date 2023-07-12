import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from sys import exit as sys_exit
from typing import Any, Optional


class Endianess(Enum):
    Big = 0
    Little = 1

    @property
    def llvm_datalayout_spec(self) -> str:
        return "E" if self == self.Big else "e"

    @classmethod
    def from_str(cls, string: str) -> "Endianess":
        return cls.Big if string == "big" else cls.Little


class ManglingStyle(Enum):
    ELF = 0

    @property
    def llvm_datalayout_spec(self) -> str:
        if self == self.ELF:
            return "m:e"
        raise NotImplementedError()

    @property
    def private_symbol_prefix(self) -> str:
        if self == self.ELF:
            return ".L"
        raise NotImplementedError()

    @classmethod
    def from_str(cls, string: str) -> "ManglingStyle":
        if string == "elf":
            return cls.ELF
        raise NotImplementedError()


class ABI(Enum):
    SystemV_AMD64 = 0

    def compute_struct_size(
        self, member_sizes: list[int], member_aligns: list[int]
    ) -> int:
        # Return the size of the struct as set by the System V AMD64 ABI.
        # [docs/abi.pdf, Section 3.1.2]

        def compute_padding(curr_size: int, align_to: int) -> int:
            # Simplifies to this, using the properties of remainders.
            return -curr_size % align_to

        # Each member is assigned to the lowest available offset with the
        # appropriate alignment.
        struct_size = 0
        for member_size, member_align in zip(member_sizes, member_aligns, strict=True):
            # Add padding to ensure each member is aligned
            struct_size += compute_padding(struct_size, member_align)

            # Append member to the struct
            struct_size += member_size

        # The size of any object is always a multiple of the object‘s alignment.
        struct_size += compute_padding(
            struct_size, self.compute_struct_alignment(member_aligns)
        )

        return struct_size

    def compute_struct_alignment(self, member_aligns: list[int]) -> int:
        # Structures and unions assume the alignment of their most strictly
        # aligned component. [docs/abi.pdf, Section 3.1.2]
        return max(member_aligns) if member_aligns else 1

    @classmethod
    def from_str(cls, string: str) -> "ABI":
        if string == "SystemV_AMD64":
            return cls.SystemV_AMD64
        raise NotImplementedError()


@dataclass
class TypeInfo:
    size: int
    align: int


@dataclass(kw_only=True)
class TargetConfig:
    arch: str
    arch_native_widths: list[int]
    abi: ABI
    endianness: Endianess
    mangling: ManglingStyle
    stack_align: int
    llvm_target_triple: str
    llvm_types: dict[str, TypeInfo]


TARGETS_DIR = Path(__file__).parent / "targets"

_target: Optional[str] = None
_target_config: Optional[TargetConfig] = None


def set_target(target: str) -> None:
    global _target, _target_config

    _target = target

    def parse_type_info(object: dict[str, Any]) -> TypeInfo | dict[str, Any]:
        # Pack into TypeInfo.
        if "size" in object and "align" in object:
            return TypeInfo(object["size"], object["align"])

        # Endianness enum.
        if "endianness" in object:
            object["endianness"] = Endianess.from_str(object["endianness"])

        # ManglingStyle enum.
        if "mangling" in object:
            object["mangling"] = ManglingStyle.from_str(object["mangling"])

        # ABI enum.
        if "abi" in object:
            object["abi"] = ABI.from_str(object["abi"])

        return object

    config_path = (TARGETS_DIR / target).with_suffix(".json")

    if not config_path.is_file():
        # TODO we should be throwing exceptions and catching them in the driver.
        print(f"Could not find a configuration file for target '{target}'")
        sys_exit(1)

    with open(config_path) as config_file:
        config_data = json.load(config_file, object_hook=parse_type_info)

    # The __init__ function checks that all specified fields are in the
    # dictionary and that no additional fields are present. This should catch
    # most mistakes/bugs in the config files and is much simpler than a formal
    # schema.
    _target_config = TargetConfig(**config_data)


def _get_config() -> TargetConfig:
    # NOTE the dataclass shouldn't be part of the public API.
    global _target_config

    assert _target_config is not None
    return _target_config


def get_datalayout() -> str:
    config = _get_config()
    specs: list[str] = []

    # Endianness.
    specs.append(config.endianness.llvm_datalayout_spec)

    # Mangling style.
    specs.append(config.mangling.llvm_datalayout_spec)

    # Pointer type.
    specs.append(f"p:{config.llvm_types['ptr'].size}:{config.llvm_types['ptr'].align}")

    # Other types.
    specs.extend(
        f"{type_name}:{type_info.align}"
        for type_name, type_info in config.llvm_types.items()
        if type_name != "ptr"
    )

    # Native integer widths.
    specs.append("n" + str.join(":", map(str, config.arch_native_widths)))

    # Alignment of the stack.
    specs.append(f"S{config.stack_align}")

    return str.join("-", specs)


def get_target_triple() -> str:
    return _get_config().llvm_target_triple


def get_llvm_type_info(llvm_type_name: str) -> TypeInfo:
    return _get_config().llvm_types[llvm_type_name]


def get_abi() -> ABI:
    return _get_config().abi
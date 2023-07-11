import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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
            return "e"
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


@dataclass
class TypeInfo:
    size: int
    align: int


@dataclass(kw_only=True)
class TargetConfig:
    arch: str
    arch_native_widths: list[int]
    env: str
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
        if "size" in object and "align" in object:
            return TypeInfo(object["size"], object["align"])
        if "endianness" in object:
            object["endianness"] = Endianess.from_str(object["endianness"])
        if "mangling" in object:
            object["mangling"] = ManglingStyle.from_str(object["mangling"])
        return object

    config_path = (TARGETS_DIR / target).with_suffix(".json")
    with open(config_path) as config_file:
        config_data = json.load(config_file, object_hook=parse_type_info)

    _target_config = TargetConfig(**config_data)


set_target("x86_64_linux")

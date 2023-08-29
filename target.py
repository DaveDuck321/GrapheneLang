import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from sys import exit as sys_exit
from sys import stderr
from typing import Optional

import yaml


class Endianess(Enum):
    BIG = 0
    LITTLE = 1

    @property
    def llvm_datalayout_spec(self) -> str:
        return "E" if self == self.BIG else "e"

    @classmethod
    def from_str(cls, string: str) -> "Endianess":
        return cls.BIG if string == "big" else cls.LITTLE


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
    SYSTEMV_AMD64 = 0
    ARM_EABI = 1
    AAPCS64 = 2

    def compute_struct_size(
        self, member_sizes: list[int], member_aligns: list[int]
    ) -> int:
        # Reuse the System V AMD64 ABI implementation for AAPCS64.
        # https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#59composite-types
        if self not in (self.SYSTEMV_AMD64, self.AAPCS64):
            raise NotImplementedError

        # Return the size of the struct as set by the System V AMD64 ABI.
        # [docs/abi.pdf, Section 3.1.2]

        def compute_padding(curr_size: int, align_to: int) -> int:
            # First, compute the number of bytes needed to pad the struct to
            # the next `align_to` boundary:
            # padding = align_to - (curr_size % align_to)
            #
            # Then, take the modulus again so that we don't add unnecessary
            # padding if the next member is already aligned:
            # return padding % align_to
            #
            # Using the properties of remainders, the above procedure simplifies
            # to:
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
        # Reuse the System V AMD64 ABI implementation for AAPCS64.
        # https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#591aggregates
        if self not in (self.SYSTEMV_AMD64, self.AAPCS64):
            raise NotImplementedError

        # Structures and unions assume the alignment of their most strictly
        # aligned component. [docs/abi.pdf, Section 3.1.2]
        return max(member_aligns) if member_aligns else 1

    def compute_stack_array_alignment(self, array_size: int, member_align: int) -> int:
        if self == self.SYSTEMV_AMD64:
            # An array uses the same alignment as its elements, except that a
            # local or global array variable of length at least 16 bytes or a
            # C99 variable-length array variable always has alignment of at
            # least 16 bytes. [docs/abi.pdf, Section 3.1.2]
            return member_align if array_size < 16 else max(member_align, 16)

        if self == self.AAPCS64:
            # The alignment of an array shall be the alignment of its base type.
            # https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#593arrays
            return member_align

        raise NotImplementedError()

    @classmethod
    def from_str(cls, string: str) -> "ABI":
        if string == "SystemV_AMD64":
            return cls.SYSTEMV_AMD64
        if string == "arm_eabi":
            # FIXME actually implement the ABI...
            return cls.ARM_EABI
        if string == "aapcs64":
            return cls.AAPCS64
        raise NotImplementedError()


@dataclass(frozen=True, slots=True)
class Size:
    in_bytes: int

    @property
    def in_bits(self) -> int:
        return self.in_bytes * 8


@dataclass(init=False, slots=True)
class TypeInfo:
    size: Size
    align: Size

    def __init__(self, size: int, align: int) -> None:
        # Both in bytes (we can't rename the arguments because they need to
        # match with the keys in the yaml files).
        self.size = Size(size)
        self.align = Size(align)


@dataclass(init=False, slots=True)
class TargetConfig:
    arch: str
    arch_native_widths: list[Size]
    abi: ABI
    endianness: Endianess
    mangling: ManglingStyle
    stack_align: Size
    int_type: TypeInfo
    llvm_target_triple: str
    llvm_types: dict[str, TypeInfo]

    def __init__(
        self,
        arch: str,
        arch_native_widths: list[int],
        abi: str,
        endianness: str,
        mangling: str,
        stack_align: int,
        int_type: dict,
        llvm_target_triple: str,
        llvm_types: dict[str, dict],
    ) -> None:
        self.arch = arch
        self.arch_native_widths = [Size(w) for w in arch_native_widths]
        self.abi = ABI.from_str(abi)
        self.endianness = Endianess.from_str(endianness)
        self.mangling = ManglingStyle.from_str(mangling)
        self.stack_align = Size(stack_align)
        self.int_type = TypeInfo(**int_type)
        self.llvm_target_triple = llvm_target_triple
        self.llvm_types = {t: TypeInfo(**d) for t, d in llvm_types.items()}


TARGETS_DIR = Path(__file__).parent / "targets"

_target: Optional[str] = None
_target_config: Optional[TargetConfig] = None


def load_target_config(target: str) -> None:
    global _target, _target_config

    # Should only call this once.
    assert _target is None
    assert _target_config is None

    _target = target

    config_path = (TARGETS_DIR / target).with_suffix(".yml")

    if not config_path.is_file():
        # TODO we should be throwing exceptions and catching them in the driver.
        print(f"Could not find a configuration file for target '{target}'", file=stderr)
        sys_exit(1)

    with open(config_path, encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    # The __init__ function checks that all specified fields are in the
    # dictionary and that no additional fields are present. This should catch
    # most mistakes/bugs in the config files and is much simpler than a formal
    # schema.
    _target_config = TargetConfig(**config_data)


def _get_config() -> TargetConfig:
    # NOTE the dataclass shouldn't be part of the public API.
    assert _target_config is not None
    return _target_config


def get_target() -> str:
    assert _target is not None
    return _target


def get_datalayout() -> str:
    config = _get_config()
    specs: list[str] = []

    # Endianness.
    specs.append(config.endianness.llvm_datalayout_spec)

    # Mangling style.
    specs.append(config.mangling.llvm_datalayout_spec)

    # Pointer type.
    specs.append(
        f"p:{config.llvm_types['ptr'].size.in_bits}:{config.llvm_types['ptr'].align.in_bits}"
    )

    # Other types.
    specs.extend(
        f"{type_name}:{type_info.align.in_bits}"
        for type_name, type_info in config.llvm_types.items()
        if type_name != "ptr"
    )

    # Native integer widths.
    specs.append(
        "n" + str.join(":", map(lambda w: str(w.in_bits), config.arch_native_widths))
    )

    # Alignment of the stack.
    specs.append(f"S{config.stack_align.in_bits}")

    return str.join("-", specs)


def get_target_triple() -> str:
    return _get_config().llvm_target_triple


def get_llvm_type_info(llvm_type_name: str) -> TypeInfo:
    return _get_config().llvm_types[llvm_type_name]


def get_ptr_type_info() -> TypeInfo:
    return get_llvm_type_info("ptr")


def get_int_type_info() -> TypeInfo:
    return _get_config().int_type


def get_abi() -> ABI:
    return _get_config().abi


def get_host_target() -> str:
    return f"{platform.machine()}_{platform.system()}".lower()

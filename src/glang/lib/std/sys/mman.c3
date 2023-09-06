// sys/mman.h - memory management declarations
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/mman.h.html

@require_once "types.c3"
@require_once "../arithmetic.c3"
@require_once "../util.c3"

// TODO: support enums (or similar)

function SYS_PROT_READ  : () -> u32 = { return 0x1; } // Page can be read.
function SYS_PROT_WRITE : () -> u32 = { return 0x2; } // Page can be written.
function SYS_PROT_EXEC  : () -> u32 = { return 0x4; } // Page can be executed.
function SYS_PROT_NONE  : () -> u32 = { return 0x0; } // Page can not be accessed.

function SYS_MAP_SHARED          : () -> u32 = { return 0x1; } // Share changes.
function SYS_MAP_PRIVATE         : () -> u32 = { return 0x2; } // Changes are private.
function SYS_MAP_SHARED_VALIDATE : () -> u32 = { return 0x3; } // Share changes and validate extension flags.

function SYS_MAP_ANONYMOUS : () -> u32 = { return 0x20; } // Don't use a file.

function mmap : (
    addr: iptr,
    length: isize,
    prot: u32,
    flags: u32,
    fd: int,
    offset: off_t
) -> iptr = {
    return sys_mmap(addr, length, as_arithmetic(prot), as_arithmetic(flags),
                    fd, offset);
}

function munmap : (addr: iptr, length: isize) -> int = {
    return Narrow<int>(sys_munmap(addr, length));
}

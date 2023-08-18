// TODO: support enums (or similar)


// https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/fcntl.h
function SYS_O_RDONLY : () -> u32 = { return 0x00; }
function SYS_O_WRONLY : () -> u32 = { return 0x01; }
function SYS_O_RDWR : () -> u32 = { return 0x02; }

// https://sites.uclouvain.be/SystInfo/usr/include/bits/mman.h.html
function SYS_PROT_READ : () -> u32 = { return 0x01; }
function SYS_PROT_WRITE : () -> u32 = { return 0x02; }
function SYS_MAP_PRIVATE : () -> u32 = { return 0x02; }
function SYS_MAP_ANONYMOUS : () -> u32 = { return 0x20; }

@require_once "../util.c3"
@require_once "../sys/uio.c3"
@require_once "../sys/types.c3"
@require_once "../fcntl.c3"

foreign _syscall_0 : (syscall_number: i32) -> i64
foreign _syscall_1 : (a: i64, syscall_number: i32) -> i64
foreign _syscall_2 : (a: i64, b: i64, syscall_number: i32) -> i64
foreign _syscall_3 : (a: i64, b: i64, c: i64, syscall_number: i32) -> i64
foreign _syscall_4 : (a: i64, b: i64, c: i64, d: i64, syscall_number: i32) -> i64
foreign _syscall_6 : (a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, syscall_number: i32) -> i64

function sys_read : (fd: int, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : iptr = __builtin_ptr_to_int(buffer);
    return _syscall_3(fd, buffer_addr, count, /* read */ 63);
}

function sys_write : (fd: int, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : iptr = __builtin_ptr_to_int(buffer);
    return _syscall_3(fd, buffer_addr, count, /* write */ 64);
}

function sys_openat : (fd: int, filename: u8[&], flags: u32, mode: u32) -> int = {
    const filename_addr : iptr = __builtin_ptr_to_int(filename);
    const result : i64 = _syscall_4(fd, filename_addr, as_arithmetic(flags),
                                    as_arithmetic(mode), /* openat */ 56);
    return Narrow<int>(result);
}

function sys_open : (filename: u8[&], flags: u32, mode: u32) -> int = {
    // There is no open syscall on aarch64 Linux. Instead, we can emulate it
    // with the newer openat syscall, with AT_FDCWD as the first parameter.
    return sys_openat(AT_FDCWD(), &filename, flags, mode);
}

function sys_close : (fd : int) -> int = {
    return Narrow<int>(_syscall_1(fd, /* close */ 57));
}

function sys_writev : (fd: int, iov: iovec[&], iovcnt: int) -> isize = {
    const iov_addr : iptr = __builtin_ptr_to_int(iov);

    return _syscall_3(fd, iov_addr, iovcnt, /* writev */ 66);
}

function sys_mmap : (
    addr: iptr,
    length: isize,
    prot: int,
    flags: int,
    fd: int,
    offset: off_t
) -> iptr = {
    return _syscall_6(addr, length, prot, flags,
                      fd, offset, /* mmap */ 222);
}

function sys_munmap : (addr: iptr, length: isize) -> i64 = {
    return _syscall_2(addr, length, /* munmap */ 215);
}

function sys_exit : (code : int) -> void = {
    _syscall_1(code, /* exit */ 93);
}

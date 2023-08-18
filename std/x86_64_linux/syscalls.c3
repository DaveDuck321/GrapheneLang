@require_once "../sys/enums.c3"
@require_once "../sys/uio.c3"
@require_once "../util.c3"

foreign _syscall_0 : (syscall_number: i64) -> i64
foreign _syscall_1 : (a: i64, syscall_number: i64) -> i64
foreign _syscall_2 : (a: i64, b: i64, syscall_number: i64) -> i64
foreign _syscall_3 : (a: i64, b: i64, c: i64, syscall_number: i64) -> i64
foreign _syscall_6 : (a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, syscall_number: i64) -> i64

function sys_read : (fd: int, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : iptr = __builtin_ptr_to_int(buffer);
    return _syscall_3(fd, buffer_addr, count, /* read */ 0);
}

function sys_write : (fd: int, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : iptr = __builtin_ptr_to_int(buffer);
    return _syscall_3(fd, buffer_addr, count, /* write */ 1);
}

function sys_open : (filename: u8[&], flags: u32, mode: u32) -> isize = {
    const filename_addr : iptr = __builtin_ptr_to_int(filename);
    return _syscall_3(filename_addr, as_arithmetic(flags), as_arithmetic(mode), /* open */ 2);
}

function sys_writev : (fd: int, iov: iovec[&], iovcnt: int) -> isize = {
    const iov_addr : iptr = __builtin_ptr_to_int(iov);
    return _syscall_3(fd, iov_addr, iovcnt, /* writev */ 20);
}

function sys_fork : () -> i64 = {
    return _syscall_0(/* fork */ 57);
}

function sys_mmap : (addr: isize, length : isize, prot: u32, flags: u32, fd: i32, offset: isize) -> iptr = {
    return _syscall_6(addr, length, as_arithmetic(prot), as_arithmetic(flags), fd, offset, 9);
}

function sys_munmap : (addr: isize, length : isize) -> int = {
    let result : isize = _syscall_2(addr, length, 11);
    return __builtin_narrow<int>(result);
}

function sys_exit : (code : int) -> void = {
    _syscall_1(code, 60);
}

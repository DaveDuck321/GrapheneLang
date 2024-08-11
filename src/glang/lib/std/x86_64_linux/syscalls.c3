@require_once "../util.c3"
@require_once "../sys/uio.c3"
@require_once "../sys/types.c3"
@require_once "sys/kstat.c3"

foreign _syscall_0 : (syscall_number: i64) -> i64
foreign _syscall_1 : (a: i64, syscall_number: i64) -> i64
foreign _syscall_2 : (a: i64, b: i64, syscall_number: i64) -> i64
foreign _syscall_3 : (a: i64, b: i64, c: i64, syscall_number: i64) -> i64
foreign _syscall_6 : (a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, syscall_number: i64) -> i64

function sys_read : (fd: int, buffer: u8[mut&], count: isize) -> isize = {
    let buffer_addr : iptr = __builtin_ptr_to_int(buffer);
    return _syscall_3(fd, buffer_addr, count, /* read */ 0);
}

function sys_write : (fd: int, buffer: u8[&], count: isize) -> isize = {
    let buffer_addr : iptr = __builtin_ptr_to_int(buffer);
    return _syscall_3(fd, buffer_addr, count, /* write */ 1);
}

function sys_open : (filename: u8[&], flags: u32, mode: u32) -> int = {
    let filename_addr : iptr = __builtin_ptr_to_int(filename);
    let result : i64 =  _syscall_3(filename_addr, as_arithmetic(flags),
                                     as_arithmetic(mode), /* open */ 2);
    return Narrow<int>(result);
}

function sys_fstat : (fd: int, out : stat_t mut&) -> int = {
    let buf_addr : iptr = __builtin_ptr_to_int(&mut out);
    return Narrow<int>(_syscall_2(fd, buf_addr, /* fstat */ 5));
}

function sys_close : (fd : int) -> int = {
    return Narrow<int>(_syscall_1(fd, /* close */ 3));
}

function sys_writev : (fd: int, iov: iovec[&], iovcnt: int) -> isize = {
    let iov_addr : iptr = __builtin_ptr_to_int(iov);
    return _syscall_3(fd, iov_addr, iovcnt, /* writev */ 20);
}

function sys_fork : () -> i64 = {
    return _syscall_0(/* fork */ 57);
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
                      fd, offset, /* mmap */ 9);
}

function sys_munmap : (addr: isize, length : isize) -> i64 = {
    return _syscall_2(addr, length, /* munmap */ 11);
}

function sys_exit : (code : int) -> void = {
    _syscall_1(code, 60);
}

function sys_memfd_create : (name : u8[&], flags : u32) -> i64 = {
    let name_addr : iptr = __builtin_ptr_to_int(&name);
    return _syscall_2(name_addr, as_arithmetic(flags), /* memfd_create */ 319);
}

function sys_ftruncate : (fd : int, length : isize) -> i64 = {
    return _syscall_2(fd, length, /* ftruncate */ 77);
}

function sys_socket : (family : int, type : int, protocol : int) -> i64 = {
    return _syscall_3(family, type, protocol, /* socket */ 41);
}

function sys_connect : (fd : int, addr : iptr, addrlen : int) -> i64 = {
    return _syscall_3(fd, addr, addrlen, /* connect */ 42);
}

function sys_pipe : (fds : int[&]) -> int = {
    return Narrow<int>(_syscall_1(__builtin_ptr_to_int(&fds), /* pipe */ 22));
}

function sys_dup2 : (old_fd : int, new_fd : int) -> int = {
    return Narrow<int>(_syscall_2(old_fd, new_fd, /* dup2 */ 33));
}

function sys_execve : (path : u8[&], argv : u8[&][&]) -> int = {
    // TODO envp.
    return Narrow<int>(_syscall_3(__builtin_ptr_to_int(&path),
        __builtin_ptr_to_int(&argv), 0, /* execve */ 59));

}

function sys_getpid : () -> int = {
    return Narrow<int>(_syscall_0(/* getpid */ 39));
}

function sys_kill : (pid : int, sig : int) -> int = {
    return Narrow<int>(_syscall_2(pid, sig, /* kill */ 62));
}

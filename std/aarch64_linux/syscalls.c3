@require_once "../sys/uio.c3"

foreign _syscall_1 : (a: i64, syscall_number: i32) -> i64
foreign _syscall_3 : (a: i64, b: i64, c: i64, syscall_number: i32) -> i64

function sys_exit : (code : int) -> void = {
    _syscall_1(code, /* exit */ 93);
}

function sys_write : (fd: int, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : iptr = __builtin_ptr_to_int(buffer);

    return _syscall_3(fd, buffer_addr, count, /* write */ 64);
}

function sys_writev : (fd: int, iov: iovec[&], iovcnt: int) -> isize = {
    const iov_addr : iptr = __builtin_ptr_to_int(iov);

    return _syscall_3(fd, iov_addr, iovcnt, /* writev */ 66);
}

foreign _syscall_1 : (a: i32, syscall_number: i32) -> i32
foreign _syscall_3 : (a: i32, b: i32, c: i32, syscall_number: i32) -> i32

function sys_exit : (code : int) -> void = {
    _syscall_1(code, /* exit */ 1);
}

function sys_write : (fd: int, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : iptr = __builtin_ptr_to_int(buffer);

    return _syscall_3(fd, buffer_addr, count, /* write */ 4);
}

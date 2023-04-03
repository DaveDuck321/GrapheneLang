foreign _syscall_0 : (syscall_number: i64) -> i64;
foreign _syscall_3 : (a: i64, b: i64, c: i64, syscall_number: i64) -> i64;

function sys_write : (fd: i32, buffer: u8[&], count: isize) -> isize = {
    const buffer_addr : i64 = __builtin_ptr_to_int<i64>(buffer);

    return _syscall_3(fd, buffer_addr, count, /* write */ 1);
};

function sys_fork : () -> i64 = {
    return _syscall_0(/* fork */ 57);
};

@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "string.c3"
@require_once "sys/uio.c3"
@require_once "syscalls.c3"
@require_once "wrappers.c3"

function EOF : () -> int = {
    // EOF - end-of-file return value
    // An integer constant expression with type int and a negative value.
    return -1;
}

function puts : (str : Span<u8>) -> int = {
    // puts - put a string on standard output
    // https://pubs.opengroup.org/onlinepubs/9699919799/functions/puts.html
    let iov : iovec[2] = {
        {&str.data, str.length},
        {"\n", 1}
    };

    const ret : isize = sys_writev(/* stdout */ 1, &iov, 2);

    if ret != -1 {
        // Upon successful completion, puts() shall return a non-negative
        // number.
        return 0;
    }

    // Otherwise, it shall return EOF,
    // TODO: shall set an error indicator for the stream,
    // TODO: and errno shall be set to indicate the error.
    return EOF();
}

function[@Len] puts : (str: u8[@Len]&) -> int = {
    return puts({&str, @Len});
}

function puts : (str: CString) -> int = {
    // TODO: why is this intermediate variable needed?
    let span : Span<u8> = {&str.data, str:length()};
    return puts(span);
}

typedef File : { fd : int }

function open : (file_path: CString, mode: u32) -> File = {
    // BUG: try removing the '&' here... the error message is nonsense
    let fd : int = __builtin_narrow<int>(sys_open(&file_path.data, mode, 0x00));
    runtime_assert(fd > 0);

    return {fd};
}

function read : (file : File&, buffer: Span<u8>) -> Span<u8> = {
    let bytes_read: isize = sys_read(file.fd, &buffer.data, buffer.length);
    return {&buffer.data, bytes_read};
}

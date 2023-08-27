@require_once "algorithms.c3"
@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "fcntl.c3"
@require_once "iterators.c3"
@require_once "math.c3"
@require_once "string.c3"
@require_once "sys/uio.c3"
@require_once "syscalls.c3"
@require_once "wrappers.c3"

typedef UnbufferedFile : { fd : int }

typedef IsStreamableTextIO<BufferedFile> : TrueType
typedef BufferedFile : {
    underlying_file : UnbufferedFile,
    buffer : u8[1024],
    buffer_pointer : isize,
}

function get_stdout_io : () -> BufferedFile = {
    let result : BufferedFile;
    result.underlying_file = {1 /* stdout */};
    result.buffer_pointer = 0;
    return result;
}

function flush : (io : BufferedFile&) -> void = {
    let data_to_write : StringView = {&io.buffer, io.buffer_pointer};
    io.underlying_file:write(data_to_write);
    io.buffer_pointer = 0;
}

function close : (io : BufferedFile&) -> void = {
    io:flush();
    io.underlying_file:close();
}

function request_buffer : (io : BufferedFile&, length : isize) -> Span<u8> = {
    runtime_assert(length <= io.buffer:length());

    if io.buffer_pointer + length < io.buffer:length() {
        // TODO: slice function
        let addr : iptr = ref_to_addr(&io.buffer[io.buffer_pointer]);
        return {&addr_to_heap_array<u8>(addr), length};
    }

    // The remaining buffer is too small, flush to make more space
    io:flush();
    return {&io.buffer, length};
}

function consume_length : (io : BufferedFile&, length : isize) -> void = {
    io.buffer_pointer += length;
}

function open : (file_path: CString, mode: u32) -> UnbufferedFile = {
    // BUG: try removing the '&' here... the error message is nonsense
    const fd : int = sys_open(&file_path.data, mode, 0x00);
    runtime_assert(fd > 0);
    return {fd};
}

function close : (file : UnbufferedFile&) -> void = {
    if file.fd >= 3 {
        // Don't close std{in,out,err}
        runtime_assert(sys_close(file.fd) > 0);
    }
}

function read : (file : UnbufferedFile&, buffer: Span<u8>) -> Span<u8> = {
    let bytes_read: isize = sys_read(file.fd, &buffer.data, buffer.length);
    return {&buffer.data, bytes_read};
}

function read : (file : BufferedFile&, buffer: Span<u8>) -> Span<u8> = {
    return file.underlying_file:read(buffer);
}

function write : (file : UnbufferedFile&, buffer : Span<u8>) -> void = {
    sys_write(file.fd, &buffer.data, buffer.length);
}

function EOF : () -> int = {
    // EOF - end-of-file return value
    // An integer constant expression with type int and a negative value.
    return -1;
}

function puts : (str : StringView) -> int = {
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
    return puts(make_span(&str));
}

function puts : (str: CString) -> int = {
    // TODO: why is this intermediate variable needed?
    let span : Span<u8> = {&str.data, str:length()};
    return puts(span);
}

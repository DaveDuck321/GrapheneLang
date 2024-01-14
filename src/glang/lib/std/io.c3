@require_once "algorithms.c3"
@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "fcntl.c3"
@require_once "iterators.c3"
@require_once "math.c3"
@require_once "string.c3"
@require_once "sys/mman.c3"
@require_once "sys/stat.c3"
@require_once "sys/uio.c3"
@require_once "syscalls.c3"
@require_once "wrappers.c3"

typedef File : { fd : int }
typedef MappedFile : { underlying : File, buffer : View<u8> }

function write : ( file : File&, buffer : View<u8> ) -> void = {
    sys_write(file.fd, &buffer.data, buffer.length);
}

function read : ( file : File&, buffer: Span<u8> ) -> View<u8> = {
    let bytes_read: isize = sys_read(file.fd, &mut buffer.data, buffer.length);
    return {&buffer.data, bytes_read};
}

function close : ( file : File& ) -> void = {
    if file.fd >= 3 {
        // Don't close std{in,out,err}
        runtime_assert(sys_close(file.fd) == 0);
    }
}

function open : ( file_path: CString, mode: u32 ) -> Optional<File> = {
    let fd : int = sys_open(&file_path.data, mode, 0x00);
    if fd < 0 {
        return make<Optional<File>>();
    }
    return make<Optional<File>>({fd});
}

function read : ( file : File& ) -> Optional<MappedFile> = {
    let stat : Optional<stat_t> = stat( file.fd );
    if !stat:has_value() {
        return make<Optional<MappedFile>>();
    }
    let size : isize = stat:data().st_size;

    let buffer : iptr = mmap(0, size, SYS_PROT_READ(), SYS_MAP_PRIVATE(), file.fd, 0);
    return make<Optional<MappedFile>>({
        file,
        { &addr_to_heap_array<u8>(buffer), size }
    });
}

function unmap : ( file : MappedFile& ) -> void = {
    munmap(ref_to_addr(&file.buffer.data), file.buffer.length);
}

typedef RegisterIsStreamableTextIO<Override<BufferedFile>> : TrueType
typedef BufferedFile : {
    underlying : File,
    buffer : u8[1024],
    buffer_pointer : isize,
}

function get_stdout_io : () -> BufferedFile = {
    mut result : BufferedFile;
    result.underlying = {1 /* stdout */};
    result.buffer_pointer = 0;
    return result;
}

function flush : (io : BufferedFile mut&) -> void = {
    let data_to_write : View<u8> = {&io.buffer, io.buffer_pointer};
    let string_to_write : StringView = {data_to_write};
    // UFCS where `write` takes a `T&` and we have a `T mut&`
    io.underlying:write(data_to_write);
    io.buffer_pointer = 0;
}

function close : (io : BufferedFile mut&) -> void = {
    io:flush();
    io.underlying:close();
}

function request_buffer : (io : BufferedFile mut&, length : isize) -> Span<u8> = {
    // TODO: this syntax for calling const length is kinda dumb
    runtime_assert(length <= io.buffer:length());

    if io.buffer_pointer + length < io.buffer:length() {
        // TODO: slice function
        let addr : iptr = ref_to_addr(&io.buffer[io.buffer_pointer]);
        return {&mut addr_to_mut_heap_array<u8>(addr), length};
    }

    // The remaining buffer is too small, flush to make more space
    io:flush();
    return {&mut io.buffer, length};
}

function consume_length : (io : BufferedFile mut&, length : isize) -> void = {
    io.buffer_pointer += length;
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
        {&str.buffer.data, str:length()},
        {"\n", 1}
    };

    let ret : isize = sys_writev(/* stdout */ 1, &iov, 2);

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
    return puts(sv(&str));
}

function puts : (str: CString) -> int = {
    return puts(make<StringView>(str));
}

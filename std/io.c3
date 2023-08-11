@require_once "algorithms.c3"
@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "iterators.c3"
@require_once "string.c3"
@require_once "sys/uio.c3"
@require_once "syscalls.c3"
@require_once "wrappers.c3"

function EOF : () -> int = {
    // EOF - end-of-file return value
    // An integer constant expression with type int and a negative value.
    return -1;
}

function print : (str : StringView) -> void = {
    // TODO: correct signature to match libc
    sys_write(/* stdout */ 1, &str.data, str.length);
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

function stringify : (buffer : u8[&], string : StringView) -> StringView = {
    return string;
}

function stringify : (buffer : u8[&], number : isize) -> StringView = {
    if number == 0 {
        buffer[0] = "0"[0];
        return {&buffer, 1};
    }

    let is_negative : bool = number < 0;
    if is_negative {
        number *= -1;
    }

    let length : isize = 0;
    while number != 0 {
        // 0 -> 48 ascii
        let next_digit: u8 = as_logical(__builtin_narrow<i8>(number % 10 + 48));

        buffer[length] = next_digit;

        number /= 10;
        length += 1;
    }

    if is_negative {
        buffer[length] = "-"[0];
        length += 1;
    }

    let result : StringView = {&buffer, length};

    // XXX: its a bit unfair to call it a `StringView` when its mutated in-place
    result:reverse();
    return result;
}

/* TODO: these overloads fail an assert
function [@Len, Args...] printf : (format_string : u8[@Len], args : Args...) -> void = {
    printf(make_span(&format_string), args...);
}

function [@Len] printf : (format_string : u8[@Len]) -> void = {
    printf(make_span(&format_string));
}
*/

function printf : (format_string : StringView) -> void = {
    print(format_string);
}

function [Args...] printf : (format_string : StringView, args : Args...) -> void = {
    let buffer : u8[1024];
    impl_printf(&buffer, format_string, args...);
}

function [Arg] impl_printf_print_until_next_format : (buffer : u8[&], format_string : StringView, next_arg : Arg) -> StringView = {
    let insertion_index : isize = format_string:index_of("%"[0]);
    if (insertion_index == -1) | (insertion_index == format_string.length - 1) {
        puts("Format string has mismatched format specifiers '%' and arguments");
        runtime_assert(false);
    }

    print({&format_string.data, insertion_index});

    // TODO: this NEEDS to be improved
    let continuation_addr : iptr = ref_to_addr(&format_string.data[insertion_index + 1]);

    // XXX: our bidmas is wrong :-|
    let next_format_string : StringView = {&addr_to_heap_array<u8>(continuation_addr), ((format_string.length - insertion_index) - 1)};

    let formatted_arg : StringView = buffer:stringify(next_arg);
    print(formatted_arg);

    return next_format_string;
}

function [NextArg] impl_printf : (buffer : u8[&], format_string : StringView, next_arg : NextArg) -> void = {
    let next_format_string : StringView = impl_printf_print_until_next_format(&buffer, format_string, next_arg);
    print(next_format_string);
}

function [NextArg, Args...] impl_printf : (buffer : u8[&], format_string : StringView, next_arg: NextArg, args : Args...) -> void = {
    let next_format_string : StringView = impl_printf_print_until_next_format(&buffer, format_string, next_arg);
    impl_printf(&buffer, next_format_string, args...);
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

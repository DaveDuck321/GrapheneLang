@require_once "io.c3"
@require_once "string.c3"
@require_once "type_traits.c3"
@require_once "util.c3"

function [IO] write : (io : IO&, char : UTF32Codepoint) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    return io:write_char(char.codepoint);
}

function [IO, @Len] write : (io : IO&, string : u8[@Len]&) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    return io:write(sv(&string));
}

function [IO] write : (io : IO&, string : StringView) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    let buffer : Span<u8> = io:request_buffer(string.length);

    // TODO: remave this assumption
    // TODO: maybe stream large strings directly
    runtime_assert(buffer.length == string.length);
    for index in range(string.length) {
        buffer.data[index] = string.data[index];
    }
    io:consume_length(buffer.length);
    return buffer;
}

function [IO] write : (io : IO&, number : isize) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    let buffer : Span<u8> = io:request_buffer(21); // len(INT_64_MIN) == 21
    if number == 0 {
        buffer.data[0] = "0"[0];
        io:consume_length(1);

        // BUG: WE CAN RETURN VOID??
        return {&buffer.data, 1};
    }

    let is_negative : bool = number < 0;
    let length : isize = 0;
    while number != 0 {
        let next_digit: u8 = as_logical(__builtin_narrow<i8>(abs(number % 10) + as_arithmetic("0"[0])));

        buffer.data[length] = next_digit;

        number /= 10;
        length += 1;
    }

    if is_negative {
        buffer.data[length] = "-"[0];
        length += 1;
    }

    let result : StringView = {&buffer.data, length};

    // XXX: its a bit unfair to call it a `StringView` when its mutated in-place
    result:reverse();
    io:consume_length(length);
    return result;
}

function [IO] write : (io : IO&, number : u32) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    // TODO: support u64
    let buffer : Span<u8> = io:request_buffer(8); // 4 bytes

    if number == 0x00 {
        buffer.data[0] = "0"[0];
        return {&buffer.data, 1};
    }

    let lookup_table : u8[16] = {
        "0"[0], "1"[0], "2"[0], "3"[0], "4"[0], "5"[0], "6"[0], "7"[0], "8"[0], "9"[0], "A"[0], "B"[0], "C"[0], "D"[0], "E"[0], "F"[0]
    };

    let length : isize = 0;
    while number != 0x0 {
        buffer.data[length] = lookup_table[as_arithmetic(number & 0xF)];

        number >>= 4;
        length += 1;
    }

    let result : StringView = {&buffer.data, length};

    // XXX: its a bit unfair to call it a `StringView` when its mutated in-place
    result:reverse();
    io:consume_length(length);
    return result;
}

function [IO] write : (io : IO&, value : bool) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    if value {
        return io:write(sv("true"));
    } else {
        return io:write(sv("false"));
    }
}

function [IO] write_char : (io : IO&, char : u8) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    let buffer : Span<u8> = io:request_buffer(1);
    buffer.data[0] = char;
    io:consume_length(1);
    return buffer;
}

function [Args...] print : (args : Args...) -> void = {
    let io : BufferedFile = get_stdout_io();
    io:print_impl(args...);
    io:write(sv("\n"));
    io:flush();
}

function [NextArg] print_impl : (io : BufferedFile&, next : NextArg) -> void = {
    io:write(next);
}

function [NextArg, OtherArgs...] print_impl : (io : BufferedFile&, next : NextArg, args : OtherArgs...) -> void = {
    io:write(next);
    io:print_impl(args...);
}

function [Args...] printf : (args : Args...) -> void = {
    let io : BufferedFile = get_stdout_io();
    io:writef(args...);
    io:write(sv("\n"));
    io:flush();
}

function [IO] writef : (io : IO&, format_string : StringView) -> void = {
    // if format_string:index_of("%"[0]) != -1 {
    //     puts("writef format string has mismatched numbers of '%' specifiers and arguments");
    //     runtime_assert(false);
    // }
    io:write(format_string);
}

function [IO, NextArg] writef : (io : IO&, format_string : StringView, next : NextArg) -> void = {
    let next_format_string : StringView = write_next_format_pair(&io, format_string, next);
    io:writef(next_format_string);
}

function [IO, NextArg, Args...] writef : (io : IO&, format_string : StringView, next_arg: NextArg, args : Args...) -> void = {
    let next_format_string : StringView = write_next_format_pair(&io, format_string, next_arg);
    io:writef(next_format_string, args...);
}

function [IO, Arg] write_next_format_pair : (io : IO&, format_string : StringView, next_arg : Arg) -> StringView = {
    let insertion_index : isize = format_string:index_of("%"[0]);
    if (insertion_index == -1) {
        // TODO: this error looks kinda janky if something has already been printed
        puts("writef format string has mismatched numbers of '%' specifiers and arguments");
        runtime_assert(false);
    }

    let format_string_view : StringView = {&format_string.data, insertion_index};
    io:write(format_string_view);
    io:write(next_arg);

    return format_string:slice_from_offset(insertion_index + 1);
}

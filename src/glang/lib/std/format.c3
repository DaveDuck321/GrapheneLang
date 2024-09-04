@require_once "io.c3"
@require_once "string.c3"
@require_once "type_traits.c3"
@require_once "util.c3"
@require_once "unicode.c3"

function [IO, @Len] write : (io : IO mut&, string : u8[@Len]&) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    io:write(sv(&string));
}

function [IO] write : (io : IO mut&, string : StringView) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    let buffer : Span<u8> = io:request_buffer(string:length_bytes());

    // TODO: remove this assumption
    // TODO: maybe stream large strings directly
    runtime_assert(buffer.length == string:length_bytes());
    for index in range(string:length_bytes()) {
        buffer.data[index] = string.buffer.data[index];
    }
    io:consume_length(buffer.length);
}

function [IO] write : (io : IO mut&, number : isize) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    let buffer : Span<u8> = io:request_buffer(21); // len(INT_64_MIN) == 21
    if number == 0 {
        buffer.data[0] = ascii_char("0");
        io:consume_length(1);

        return;
    }

    let is_negative : bool = number < 0;
    mut current_number : isize = number;
    mut length : isize = 0;
    while current_number != 0 {
        let next_digit: u8 = as_logical(__builtin_narrow<i8>(abs(current_number % 10) + as_arithmetic(ascii_char("0"))));

        buffer.data[length] = next_digit;

        current_number /= 10;
        length += 1;
    }

    if is_negative {
        buffer.data[length] = ascii_char("-");
        length += 1;
    }

    let result : Span<u8> = make<Span<u8>>(&mut buffer.data, length);
    result:reverse();
    io:consume_length(length);
}

function [IO] write : (io : IO mut&, number : u32) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    // TODO: support u64
    let buffer : Span<u8> = io:request_buffer(8); // 4 bytes

    if number == 0x00 {
        buffer.data[0] = ascii_char("0");
        io:consume_length(1);
        return;
    }

    let lookup_table : u8[16] = {
        ascii_char("0"),
        ascii_char("1"),
        ascii_char("2"),
        ascii_char("3"),
        ascii_char("4"),
        ascii_char("5"),
        ascii_char("6"),
        ascii_char("7"),
        ascii_char("8"),
        ascii_char("9"),
        ascii_char("A"),
        ascii_char("B"),
        ascii_char("C"),
        ascii_char("D"),
        ascii_char("E"),
        ascii_char("F")
    };

    mut remaining_number : u32 = number;
    mut length : isize = 0;
    while remaining_number != 0x0 {
        buffer.data[length] = lookup_table[as_arithmetic(remaining_number & 0xF)];

        remaining_number >>= 4;
        length += 1;
    }

    let result : Span<u8> = buffer:slice_to(length);
    result:reverse();
    io:consume_length(length);
}

function [IO] write : (io : IO mut&, value : bool) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    if value {
        io:write(sv("true"));
    } else {
        io:write(sv("false"));
    }
}

function [IO] write_byte : (io : IO mut&, char : u8) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    let buffer : Span<u8> = io:request_buffer(1);
    buffer.data[0] = char;
    io:consume_length(1);
}

function [Args...] print : (args : Args...) -> void = {
    mut io : BufferedFile = get_stdout_io();
    io:print_impl(args...);
    io:write(sv("\n"));
    io:flush();
}

function [NextArg] print_impl : (io : BufferedFile mut&, next : NextArg) -> void = {
    io:write(next);
}

function [NextArg, OtherArgs...] print_impl : (io : BufferedFile mut&, next : NextArg, args : OtherArgs...) -> void = {
    io:write(next);
    io:print_impl(args...);
}

function [Args...] printf : (args : Args...) -> void = {
    mut io : BufferedFile = get_stdout_io();
    io:writef(args...);
    io:write(sv("\n"));
    io:flush();
}

function [IO] writef : (io : IO mut&, format_string : StringView) -> void = {
    if index_of(format_string:get_iter(), char("%")) != -1 {
        puts("writef format string has mismatched numbers of '%' specifiers and arguments");
        runtime_assert(false);
    }
    io:write(format_string);
}

function [IO, NextArg] writef : (io : IO mut&, format_string : StringView, next : NextArg) -> void = {
    let next_format_string : StringView = write_next_format_pair(&mut io, format_string, next);
    io:writef(next_format_string);
}

function [IO, NextArg, Args...] writef : (io : IO mut&, format_string : StringView, next_arg: NextArg, args : Args...) -> void = {
    let next_format_string : StringView = write_next_format_pair(&mut io, format_string, next_arg);
    io:writef(next_format_string, args...);
}

function [IO, Arg] write_next_format_pair : (io : IO mut&, format_string : StringView, next_arg : Arg) -> StringView = {
    let insertion_index : isize = index_of(format_string:get_iter(), char("%"));
    if (insertion_index == -1) {
        // TODO: this error looks kinda janky if something has already been printed
        puts("writef format string has mismatched numbers of '%' specifiers and arguments");
        runtime_assert(false);
    }

    let format_string_view : StringView = {format_string.buffer:slice_to(insertion_index)};
    io:write(format_string_view);
    io:write(next_arg);

    return {format_string.buffer:slice_from(insertion_index + 1)};
}

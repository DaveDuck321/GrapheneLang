@require_once "std/io.c3"

function main : () -> int = {
    const str: u8[&] = "abc";
    puts(&const str);
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: overload resolution for function call 'puts(u8[[]&[]])' failed
/// Available overloads:
/// - function puts: (StringView) -> int
/// - function puts: (CString) -> int

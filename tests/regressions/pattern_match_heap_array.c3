@require_once "std/io.c3"

function main : () -> int = {
    let str: u8[&] = "abc";
    puts(&str);
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: overload resolution for function call 'puts(u8[[]&[]])' failed
/// Available overloads:
/// - function [[]@Len[]] puts<@Len> : (u8[[]@Len[]]&) -> int
/// - function puts : (CString) -> int
/// - function puts : (StringView) -> int

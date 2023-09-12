function fn : (a : u8[&]) -> int = {}

function fn_launcher : (a : u8[&]) -> int = {
    return fn(a);
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 4, in 'function fn_launcher: (u8[[]&[]]) -> int'
/// Error: overload resolution for function call 'fn(u8[[]&[]] (unborrowed))' failed
/// Available overloads:
/// - function fn : (u8[[]&[]]) -> int

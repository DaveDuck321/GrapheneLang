function [T] takes_ref : (a : T&) -> void = {}

function main : () -> int = {
    let array : u8[&] = "hello";
    takes_ref(&array);
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: overload resolution for function call 'takes_ref(u8[[]&[]])' failed
/// Available overloads:
/// - function [[]T[]] takes_ref<T> : (T&) -> void

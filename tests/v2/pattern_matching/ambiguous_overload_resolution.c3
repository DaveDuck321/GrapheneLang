typedef S : {a : int, b : int};

function[T] get : (x : {a : T, b : T}) -> void = {};

function[T] get : (x : T) -> void = {};

function main : () -> int = {
    const s : S = {1, 2};
    get(s);

    return 0;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 9, in 'function main: () -> int'
/// Error: overload resolution for function call 'get(S)' is ambiguous.
/// Equally good candidates are
/// - function get<int>: ({a : int, b : int}) -> void
/// - function get<S>: (S) -> void

function takes_ref : (a : int&) -> void = {}

function main : () -> int = {
    mut a : int = 7;
    takes_ref(&mut a);
    return a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: overload resolution for function call 'takes_ref(int mut&)' failed
/// Available overloads:
/// - function takes_ref : (int&) -> void

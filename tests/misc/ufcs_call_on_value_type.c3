typedef A : {}

function gives_value : () -> A = {
    return {};
}

function fn : (a : A&) -> void = {}

function main: () -> int = {
    gives_value():fn();
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 10, in 'function main: () -> int'
/// Error: cannot borrow type 'A' (with no address)

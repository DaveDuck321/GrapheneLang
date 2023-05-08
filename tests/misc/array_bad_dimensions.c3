typedef A : int[6, 0];

function main: () -> int = {
    return a(0, 0);
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'typedef'
/// Error: cannot construct array type 'int[[]6, 0[]]' since it has a non-positive size

typedef my_struct: {a: int}

function main: () -> int = {
    const s : my_struct = {.b = 2};

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 4, in 'function main: () -> int'
/// Error: initializer list of the form '{b: int}' cannot be converted to 'typedef my_struct = {a: int}'

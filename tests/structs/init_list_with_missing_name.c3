typedef my_struct: {a: int, b: int}

function main: () -> int = {
    const s : my_struct = { .b = 2 };

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 4, in 'function main: () -> int'
/// Error: initializer list with 1 object cannot be converted to a struct with 2 members

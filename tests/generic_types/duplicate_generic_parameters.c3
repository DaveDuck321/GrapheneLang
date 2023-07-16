typedef[T, T] my_struct : {a: T, b: T}

function main: () -> int = {
    // Linker fails without a main().
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'typedef my_struct : ...'
/// Error: generic 'T' appears more than once in the declaration of 'my_struct'

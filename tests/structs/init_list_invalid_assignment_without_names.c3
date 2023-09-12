function main: () -> int = {
    let not_a_struct : int = {2, "str", true};

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: initializer list of the form '{int, u8[[]3[]]&, bool}' cannot be converted to 'int'

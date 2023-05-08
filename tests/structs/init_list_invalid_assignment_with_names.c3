function main: () -> int = {
    const not_a_struct : int = {a: 2, b: "str", c: true};

    return 0;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: initializer list of the form '{a: int, b: u8[[]3[]]&, c: bool}' cannot be converted to 'int'

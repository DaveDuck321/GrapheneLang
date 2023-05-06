function main : () -> int = {
    let x : void;

    return 0;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: variable 'x' cannot have type 'void'

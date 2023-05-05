typedef my_void : void;
typedef my_struct : { a: int, b: my_void };

function main : () -> int = {
    let s : my_struct;

    return 0;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: struct member 'b' cannot have type 'typedef my_void = void'

typedef my_void : void
typedef my_struct : { a: int, b: my_void }

function main : () -> int = {
    let s : my_struct;

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'typedef'
/// Error: struct '{a: int, b: my_void}' cannot have member 'b' of type 'typedef my_void = void'

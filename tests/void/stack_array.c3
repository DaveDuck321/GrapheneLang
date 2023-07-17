typedef my_void : void
typedef my_array : my_void[6]

function main : () -> int = {
    let s : my_struct;

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'typedef'
/// Error: array 'my_void[[]6[]]' cannot operate on scalar 'typedef my_void = void'

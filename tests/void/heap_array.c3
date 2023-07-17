typedef my_void : void
typedef my_array : my_void[&]

function main : () -> int = {
    let s : my_struct;

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'typedef'
/// Error: array 'my_void[[]&[]]' cannot operate on scalar 'typedef my_void = void'

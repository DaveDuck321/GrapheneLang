typedef my_int : i64;

function main : () -> my_int = {
    return 0;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function signature'
/// Error: type 'typedef my_int = i64' is not a valid return type for function 'main'; expected 'int'

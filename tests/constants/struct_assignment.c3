typedef MyStruct : {a : int}

function main : () -> int = {
    let struct : MyStruct = {1};
    struct.a = 1;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: left hand side of assignment operation has non-mutable type 'int&'

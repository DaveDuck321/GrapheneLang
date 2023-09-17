typedef MyStruct : {a : int&}

function main : () -> int = {
    mut a : int = 7;
    let struct : MyStruct = {&a};
    struct.a = 0;
    return a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 6, in 'function main: () -> int'
/// Error: left hand side of assignment operation has non-mutable type 'int&'

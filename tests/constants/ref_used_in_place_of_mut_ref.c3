typedef MyStruct : {a : int mut&}

function main : () -> int = {
    mut a : int = 7;
    let struct : MyStruct = {&a};
    struct.a = 0;
    return a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error in struct initializer: type 'int&' does not match expected type 'int mut&'

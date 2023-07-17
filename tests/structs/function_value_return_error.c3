typedef MyStruct : {a: int, x: int}

function make_struct : () -> MyStruct = {
    let ret : MyStruct = {42, 10};
    return ret;
}

function main : () -> int = {
    make_struct().x = 10;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 9, in 'function main: () -> int'
/// Error: cannot modify temporary struct

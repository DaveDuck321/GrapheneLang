typedef MyStruct : {a : int mut&}

function main : () -> int = {
    mut a : int = 7;
    let struct : MyStruct = {&mut a};
    struct.a = 0;
    return a;
}

/// @COMPILE
/// @RUN

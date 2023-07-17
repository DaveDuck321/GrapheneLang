typedef MyStruct : {a: int, x: int}

function make_struct : () -> MyStruct = {
    let ret : MyStruct = {42, 10};
    return ret;
}

function main : () -> int = {
    return make_struct().x;
}

/// @COMPILE
/// @RUN; EXPECT 10

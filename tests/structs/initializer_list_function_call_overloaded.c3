typedef MyStruct1 : {a: int, x: int, b: int}
typedef MyStruct2 : {a: int, x: int}

function get_x : (struct : MyStruct1) -> int = {
    return struct.x;
}

function get_x : (struct : MyStruct2) -> int = {
    return struct.x;
}

function main : () -> int = {
    return get_x({11, 12});
}

/// @COMPILE
/// @RUN; EXPECT 12

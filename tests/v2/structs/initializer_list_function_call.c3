typedef MyStruct : {a: int, x: int};

function get_x : (struct : MyStruct) -> int = {
    return struct.x;
};

function main : () -> int = {
    return get_x({11, 12});
};

/// @COMPILE
/// @RUN; EXPECT 12

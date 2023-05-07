typedef MyStruct : {a: int, x: int};

function get_struct : () -> MyStruct = {
    return {1, 2};
};

function main : () -> int = {
    return get_struct().a;
};

/// @COMPILE
/// @RUN; EXPECT 1

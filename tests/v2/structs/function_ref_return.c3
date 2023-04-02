typedef MyStruct : {a: int, x: int};

function[T] forward_ref : (arg: T&) -> T& = {
    return &arg;
};

function main : () -> int = {
    let struct : MyStruct = {42, 10};
    forward_ref<MyStruct>(&struct).x = 11;

    return forward_ref<MyStruct>(&struct).x;
};

/// @COMPILE
/// @RUN; EXPECT 11

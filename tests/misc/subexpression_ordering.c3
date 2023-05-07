foreign puts : (str: u8[&]) -> int;

function a : () -> int = {
    puts("a");
    return 1;
};

function b : (b: int) -> int = {
    puts("b");
    return 7;
};

function c : () -> int = {
    puts("c");
    return 1;
};

function main: () -> int = {
    b(c());
    c();
    return a();
};

/// @COMPILE
/// @RUN; EXPECT 1; EXPECT OUT
/// c
/// b
/// c
/// a

foreign puts : (str: u8[&]) -> int

function a : () -> int = {
    puts("a\0");
    return 1;
}

function b : (b: int) -> int = {
    puts("b\0");
    return 7;
}

function c : () -> int = {
    puts("c\0");
    return 1;
}

function main: () -> int = {
    b(c());
    c();
    return a();
}

/// @COMPILE --use-crt
/// @RUN; EXPECT 1; EXPECT OUT
/// c
/// b
/// c
/// a

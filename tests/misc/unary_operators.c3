foreign puts : (str: u8[&]) -> int

function a : () -> int = {
    puts("a\0");
    return 1;
}

@operator - : (rhs: int) -> int = {
    puts("-\0");
    return __builtin_add(rhs, 6);
}

function main: () -> int = {
    return -a();
}

/// @COMPILE --use-crt
/// @RUN; EXPECT 7; EXPECT OUT
/// a
/// -

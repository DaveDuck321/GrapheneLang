foreign puts : (str: u8[&]) -> int

@operator + : (rhs: int, lhs: int) -> int = {
    puts("+\0");
    return 0;
}

@operator * : (rhs: int, lhs: int) -> int = {
    puts("*\0");
    return 0;
}

@operator | : (rhs: int, lhs: int) -> bool = {
    puts("|\0");
    return false;
}

@operator ** : (rhs: int, lhs: int) -> int = {
    puts("**\0");
    return 0;
}

@operator == : (rhs: int, lhs: int) -> bool = {
    puts("==\0");
    return false;
}

function main : () -> int = {
    let a: int = 1 + 2 * 3;
    let b: int = 2 * 3 + 1;
    let c: bool = 1 + 2 | 3 + 4;
    let d: int = 1 * 2 ** 8 * 3;
    let e: bool = 1 * 2 == 8 * 3;
    return 0;
}

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// *
/// +
/// *
/// +
/// +
/// +
/// |
/// **
/// *
/// *
/// *
/// *
/// ==

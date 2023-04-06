@require_once "std/io.c3";

@operator + : (rhs: int, lhs: int) -> int = {
    puts("+");
    return 0;
};

@operator * : (rhs: int, lhs: int) -> int = {
    puts("*");
    return 0;
};

@operator || : (rhs: int, lhs: int) -> bool = {
    puts("||");
    return false;
};

@operator ** : (rhs: int, lhs: int) -> int = {
    puts("**");
    return 0;
};

@operator == : (rhs: int, lhs: int) -> bool = {
    puts("==");
    return false;
};

function main : () -> int = {
    const a: int = 1 + 2 * 3;
    const b: int = 2 * 3 + 1;
    const c: bool = 1 + 2 || 3 + 4;
    const d: int = 1 * 2 ** 8 * 3;
    const e: bool = 1 * 2 == 8 * 3;
    return 0;
};

/// @COMPILE
/// @RUN; EXPECT OUT
/// *
/// +
/// *
/// +
/// +
/// +
/// ||
/// **
/// *
/// *
/// *
/// *
/// ==

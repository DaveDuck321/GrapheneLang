foreign puts : (str: u8[&]) -> int;

@operator - : (val: int) -> int = {
    // unary_operator: highest precedence.
    puts("-");
    return 0;
};

@operator ** : (rhs: int, lhs: int) -> int = {
    // power.
    puts("**");
    return 0;
};

@operator * : (rhs: int, lhs: int) -> int = {
    // mult_div.
    puts("*");
    return 0;
};

@operator + : (rhs: int, lhs: int) -> int = {
    // add_sub.
    puts("+");
    return 0;
};

@operator >> : (rhs: int, lhs: int) -> int = {
    // bit_shift.
    puts(">>");
    return 0;
};

@operator & : (rhs: int, lhs: int) -> int = {
    // bitwise_and.
    puts("&");
    return 0;
};

@operator ^ : (rhs: int, lhs: int) -> int = {
    // bitwise_xor.
    puts("^");
    return 0;
};

@operator | : (rhs: int, lhs: int) -> int = {
    // bitwise_or.
    puts("|");
    return 0;
};

@operator == : (rhs: int, lhs: int) -> bool = {
    // comparison_op.
    puts("==");
    return false;
};

function main : () -> int = {
    const a: int = 0;
    const r: bool = -a**5 * 2 >> 3 + 2 ^ 1 | 1 & 0 == 0;
    return 0;
};

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// -
/// **
/// *
/// +
/// >>
/// ^
/// &
/// |
/// ==

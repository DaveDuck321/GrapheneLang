@operator == : (lhs: int, rhs: int) -> bool = {
    return __builtin_is_equal(lhs, rhs);
};

@operator != : (lhs: int, rhs: int) -> bool = {
    if (lhs == rhs) {
        return false;
    }
    return true;
};

@operator < : (lhs: int, rhs: int) -> bool = {
    return __builtin_is_less_than(lhs, rhs);
};

@operator > : (lhs: int, rhs: int) -> bool = {
    return __builtin_is_greater_than(lhs, rhs);
};

@operator * : (lhs: int, rhs: int) -> int = {
    return __builtin_multiply(lhs, rhs);
};

@operator / : (lhs: int, rhs: int) -> int = {
    return __builtin_divide(lhs, rhs);
};

@operator - : (lhs: int, rhs: int) -> int = {
    return __builtin_subtract(lhs, rhs);
};

@operator + : (lhs: int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

@operator >> : (lhs: int, rhs: int) -> int = {
    return __builtin_shift_right(lhs, rhs);
};

@operator << : (lhs: int, rhs: int) -> int = {
    return __builtin_shift_left(lhs, rhs);
};

@operator & : (lhs: int, rhs: int) -> int = {
    return __builtin_bitwise_and(lhs, rhs);
};

@operator | : (lhs: int, rhs: int) -> int = {
    return __builtin_bitwise_or(lhs, rhs);
};

@operator ^ : (lhs: int, rhs: int) -> int = {
    return __builtin_bitwise_xor(lhs, rhs);
};

foreign puts : (str: u8[&]) -> int;

function main : () -> int = {
    if (1 + 1 != 3) {
        puts("sanity check\0");
    }

    // All of the following check for obvious errors
    if (2 != 2) {
        return 1;
    }
    if (1 > 2) {
        return 1;
    }
    if (-1 > 2) {
        return 1;
    }

    if (2 < 1) {
        return 1;
    }
    if (2 < -1) {
        return 1;
    }

    if (1 + 1 != 2) {
        return 1;
    }
    if (-1 + 1 != 0) {
        return 1;
    }

    if (3*4 != 12) {
        return 1;
    }
    if (3*-4 != -12) {
        return 1;
    }

    if (10 / 4 != 2) {
        return 1;
    }
    if (-10 / 4 != -2) {
        return 1;
    }

    if (9 >> 1 != 4) {
        return 1;
    }
    if (-2 >> 1 != -1) {
        return 1;
    }
    if (9 << 1 != 18) {
        return 1;
    }

    if (5 & 4 != 4) {
        return 1;
    }
    if (5 | 4 != 5) {
        return 1;
    }
    if (5 ^ 4 != 1) {
        return 1;
    }
    return 0;
};

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// sanity check

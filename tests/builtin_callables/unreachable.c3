function main : () -> int = {
    // Cannot be executed, use opt to validate ir

    __builtin_unreachable();
    // Note: no return here
}

/// @COMPILE
/// @OPTIMIZE
/// @GREP_IR unreachable

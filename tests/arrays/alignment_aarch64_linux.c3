function main : () -> int = {
    // Arrays use the alignment of their elements.
    mut small : int[3];

    // Irrespective of the total array size.
    mut large : int[4];

    return 0;
}

/// @FOR aarch64_linux
/// @COMPILE
/// @GREP_IR [3 x i32], align 4
/// @GREP_IR [4 x i32], align 4

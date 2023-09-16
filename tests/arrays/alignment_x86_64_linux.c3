function main : () -> int = {
    // Arrays use the alignment of their elements.
    mut small : int[3];

    // Except if their length is at least 16 bytes, in which case the alignment
    // should be at least 16.
    mut large : int[4];

    return 0;
}

/// @FOR x86_64_linux
/// @COMPILE
/// @GREP_IR [3 x i32], align 4
/// @GREP_IR [4 x i32], align 16

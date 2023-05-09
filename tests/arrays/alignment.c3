function main : () -> int = {
    // Arrays use the alignment of their elements.
    let small : int[3];

    // Except if their length is at least 16 bytes, in which case the alignment
    // should be at least 16.
    let large : int[4];

    return 0;
};

/// @COMPILE
/// @GREP_IR [3 x i32], align 4
/// @GREP_IR [4 x i32], align 16

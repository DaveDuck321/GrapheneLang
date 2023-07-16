typedef MyStruct1 : {a : i32, b : i8, c : i16, d : i8}
typedef MyStruct2 : {a : i8, b : i16, c : i8, d : i32}
typedef MyArray1 : i32[5]
typedef MyArray2 : i8[5]

@operator != : (lhs : isize, rhs : isize) -> bool = {
    if (__builtin_is_equal(lhs, rhs)) {
        return false;
    }
    return true;
}

function main: () -> int = {
    // TODO: we get an uninitialized variable warning without the =0
    let align : isize = 0;

    align = __builtin_alignof<i16>();
    if (align != 2) {
        return 1;
    }

    align = __builtin_alignof<MyStruct1>();
    if (align != 4) {
        return 1;
    }

    align = __builtin_alignof<MyStruct2>();
    if (align != 4) {
        return 1;
    }

    align = __builtin_alignof<MyArray1>();
    // Arrays of length at least 16 bytes have an alignment of at least 16
    // bytes.
    if (align != 16) {
        return 1;
    }

    align = __builtin_alignof<MyArray2>();
    if (align != 1) {
        return 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN

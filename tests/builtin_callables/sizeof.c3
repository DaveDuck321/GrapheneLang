typedef MyStruct1 : {a : i32, b : i8, c : i8, d : i16} // Elements ordered to avoid padding
typedef MyStruct2 : {a : i8, b : i16, c : i32, d : i8} // Very poor padding
typedef MyArray1 : i32[5]
typedef MyArray2 : i8[5]

@operator != : (lhs : isize, rhs : isize) -> bool = {
    if __builtin_is_equal(lhs, rhs) {
        return false;
    }
    return true;
}

function main: () -> int = {
    // TODO: we get an uninitialized variable warning without the =0
    let size : isize = 0;

    size = __builtin_sizeof<i16>();
    if size != 2 {
        return 1;
    }

    size = __builtin_sizeof<MyStruct1>();
    if size != 8 {
        return 1;
    }

    size = __builtin_sizeof<MyStruct2>();
    if size != 12 {
        return 1;
    }

    size = __builtin_sizeof<MyArray1>();
    if size != 20 {
        return 1;
    }

    size = __builtin_sizeof<MyArray2>();
    if size != 5 {
        return 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN

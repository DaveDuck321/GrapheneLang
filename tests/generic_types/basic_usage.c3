typedef[T, U, V] my_struct : {a: T, b: U, c: V}

function main: () -> int = {
    mut s : my_struct<i32, i16, i8>;

    return 0;
}

/// @COMPILE
/// @GREP_IR %type._Z9my_structIiscE = type {i32, i16, i8}
/// @RUN

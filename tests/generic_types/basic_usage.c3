typedef[T, U, V] my_struct : {a: T, b: U, c: V}

function main: () -> int = {
    mut s : my_struct<i32, i16, i8>;

    return 0;
}

/// @COMPILE
/// @GREP_IR %type.__T_my_struct__S__T_i32__T_i16__T_i8 = type {i32, i16, i8}
/// @RUN

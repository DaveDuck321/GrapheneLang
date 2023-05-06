typedef[T, U, V] my_struct : {a: T, b: U, c: V};

function main: () -> int = {
    let s : my_struct<i32, i16, i8>;

    return 0;
};

/// @COMPILE
/// @RUN

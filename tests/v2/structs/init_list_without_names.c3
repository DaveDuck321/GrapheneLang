typedef my_i32 : int;
typedef[T] my_struct : {a: T, b: my_i32};

@operator +: (a: i64, b: i64) -> i64 = {
    return __builtin_add(a, b);
};

function main: () -> int = {
    let x : int = 1;
    let y : int = 2;

    let s1 : my_struct<int> = {x, y};
    let s2 : my_struct<int> = {10, 20};

    let s3 : my_struct<i64> = {x, y};
    let s4 : my_struct<i64> = {10, 20};

    return __builtin_narrow<int>(s1.a + s2.b + s3.b + s4.a);
};

/// @COMPILE
/// @RUN; EXPECT 33

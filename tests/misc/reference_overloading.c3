@require_once "std/io.c3"

typedef MyStruct : {a : int}

function fn : (a: int) -> int = {
    puts("val");
    return 0;
}

function fn : (a: int mut&) -> int = {
    puts("ref");
    return 0;
}

function main: () -> int = {
    mut x: int = 0;
    mut my_struct : MyStruct;
    fn(x);
    fn(&mut x);
    fn(0);
    fn(&mut my_struct.a);
    fn(my_struct.a);
    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// val
/// ref
/// val
/// ref
/// val

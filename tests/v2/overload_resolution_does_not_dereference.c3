@require_once "std/io.c3";

function f: (x: int, y: int) -> int = { puts("val"); return 0; };
function f: (x: int&, y: i64) -> int = { puts("ref"); return 0; };

function main: () -> int = {
    let x: int = 4;
    let y: int = 8;

    f(x, y);
    f(&x, y);

    return 0;
};

/// @COMPILE
/// @RUN; EXPECT OUT
/// val
/// ref

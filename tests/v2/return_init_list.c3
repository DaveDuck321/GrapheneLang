@require_once "std/arithmetic.c3";

typedef S: { a: int, b: int };

function foo: () -> S = {
    return { 1, 2 };
};

function main: () -> int = {
    const s: S = foo();

    return s.a + s.b;
};

/// @COMPILE
/// @RUN; EXPECT 3

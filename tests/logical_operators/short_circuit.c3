@require_once "std/io.c3"
@require_once "std/logical.c3"

function a : () -> bool = {
    puts("a");
    return true;
}

function b : () -> bool = {
    puts("b");
    return false;
}

function c : () -> bool = {
    puts("c");
    return true;
}

function main : () -> int = {
    // Should print a, b.
    a() and b() and c();

    // Should print b, a.
    b() or a() or c();

    // Should print b, c. Note that and has higher precedence.
    b() and a() or c();

    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// a
/// b
/// b
/// a
/// b
/// c

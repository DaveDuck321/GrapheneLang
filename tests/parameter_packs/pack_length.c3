@require_once "std/arithmetic.c3"

typedef S : {}
typedef T : {}

function[T] pack_len: (curr: T) -> int = {
    return 1;
}

function[T, Ys...] pack_len: (curr: T, next: Ys...) -> int = {
    return pack_len(next...) + 1;
}

function main : () -> int = {
    const s : S = {};
    const t : T = {};

    return pack_len(1, 2, 3, s, t);
}

/// @COMPILE
/// @RUN; EXPECT 5

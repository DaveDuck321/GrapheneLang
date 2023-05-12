@require_once "std/arithmetic.c3";

function[T] pack_len: (curr: T) -> int = {
    // Base case
    return 1;
};

function[T, Ys...] pack_len: (curr: T, next: Ys...) -> int = {
    // Very Haskell
    return pack_len(next...) + 1;
};

// FIXME format should be a base/length string type.
function[@Len, T, Ys...] printf: (format: u8[@Len], val: T, next_vals: Ys...) -> void = {

};

function main : () -> int = {
    return pack_len(1, 2, 3);
};

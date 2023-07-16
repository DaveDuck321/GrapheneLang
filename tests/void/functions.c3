@operator > : (a : int, b: int) -> bool = {
    return __builtin_is_greater_than(a, b);
}

function foo : (i : int&) -> void = {
    if (i > 2) {
        return;
    }

    let x : int;

    // We need to add an implicit `ret void` here or LLVM will complain.
}

function main : () -> int = {
    let i : int = 1;
    foo(&i);

    return 0;
}

/// @COMPILE
/// @RUN

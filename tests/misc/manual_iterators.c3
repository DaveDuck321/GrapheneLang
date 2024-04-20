@require_once "std/arithmetic.c3"

typedef MyType : {val : int}

@implicit has_next : (a : MyType&) -> bool = {
    return true;
}

@implicit get_next : (a : MyType mut&) -> int = {
    a.val += 1;
    return a.val;
}

function main : () -> int = {
    mut a : MyType = {0};
    __builtin_get_next(&mut a);
    __builtin_get_next(&mut a);
    __builtin_get_next(&mut a);
    __builtin_get_next(&mut a);

    if not __builtin_has_next(&a) {
        return 0;
    }
    return a.val;
}

/// @COMPILE
/// @RUN; EXPECT 4

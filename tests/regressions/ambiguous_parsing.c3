typedef [T] A : { a : T }
typedef [T] B : { b : T }

function [T] foo : () -> T = {
    return {{2}};
}

@operator - : (value : int) -> int = {
    return value;
}

@operator < : (a : int, b: int) -> bool = {
    return true;
}

function main : () -> int = {
    let val : A<B<int>> = foo<A<B<int>>>();

    if 0 < 1 {
        return -val.a.b;
    } else {
        return 1;
    }

    // FIXME https://github.com/DaveDuck321/GrapheneLang/issues/72
    return 0;
}

/// @COMPILE
/// @RUN; EXPECT 2

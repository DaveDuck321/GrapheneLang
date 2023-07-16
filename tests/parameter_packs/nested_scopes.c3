function[Ts...] inner: (args: Ts...) -> int = {
    return 0;
}

function[T, Ys...] outer: (curr: T, next: Ys...) -> int = {
    if (true) {
        return inner(next...);
    } else {
        return 1;
    }
}

function main : () -> int = {
    return outer(1, 2, 3);
}

/// @COMPILE
/// @RUN

function foo : () -> void = {
    if true {
        return;
    } else {
        return;
    }
    return;
}

function main : () -> int = {
    foo();
    return 0;
}

/// @COMPILE
/// @RUN

function foo : (a : int[2, 3]) -> int = {
    return 0;
}

function main : () -> int = {
    // FIXME call foo() once we support array constants.
    return 0;
}

/// @COMPILE
/// @RUN

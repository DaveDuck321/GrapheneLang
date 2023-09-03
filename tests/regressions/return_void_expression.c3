function foo : () -> void = {}
function bar : () -> void = { return foo(); }

function main: () -> int = {
    bar();
    return 0;
}

/// @COMPILE
/// @RUN

function truthy : () -> bool = { return true; }
function falsy : () -> bool = { return false; }

function main : () -> int = {
    if falsy() and (truthy() or falsy()) {
        return 1;
    }
    return 0;
}

/// @COMPILE
/// @RUN

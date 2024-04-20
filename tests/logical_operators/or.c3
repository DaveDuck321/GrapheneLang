@require_once "std/logical.c3"

function truthy : () -> bool = { return true; }
function falsy : () -> bool = { return false; }

function main : () -> int = {
    if falsy() or falsy() {
        return 1;
    }
    if not (truthy() or falsy()) {
        return 1;
    }
    if not (falsy() or truthy()) {
        return 1;
    }
    if not (truthy() or truthy()) {
        return 1;
    }
    return 0;
}

/// @COMPILE
/// @RUN

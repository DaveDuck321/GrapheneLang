@require_once "std/io.c3"
@require_once "std/logical.c3"

function truthy : () -> bool = { return true; }
function falsy : () -> bool = { return false; }

function main : () -> int = {
    if falsy() and falsy() {
        return 1;
    }
    if truthy() and falsy() {
        return 1;
    }
    if falsy() and truthy() {
        return 1;
    }
    if !(truthy() and truthy()) {
        return 1;
    }
    return 0;
}

/// @COMPILE
/// @RUN

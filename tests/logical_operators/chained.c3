@require_once "std/io.c3"
@require_once "std/logical.c3"

function truthy : () -> bool = { return true; }
function falsy : () -> bool = { return false; }

function main : () -> int = {
    if falsy() and falsy() and falsy() {
        return 1;
    }
    if truthy() and falsy() and falsy() {
        return 1;
    }
    if falsy() and truthy() and truthy() {
        return 1;
    }
    if not (truthy() and truthy() and truthy()) {
        return 1;
    }

    if falsy() or falsy() or falsy() {
        return 1;
    }
    if not (truthy() or falsy() or falsy()) {
        return 1;
    }
    if not (falsy() or truthy() or truthy()) {
        return 1;
    }
    if not (truthy() or truthy() or truthy()) {
        return 1;
    }

    // or has lower precedence.
    if not (falsy() and falsy() or truthy()) {
        return 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN

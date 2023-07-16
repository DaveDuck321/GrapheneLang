@require_once "std/logical.c3"
@require_once "std/type_traits.c3"

function main : () -> int = {
    let a : bool = true;
    let b : bool = false;

    if (a & b != false) {
        return 1;
    }
    if (a | b != true) {
        return 1;
    }
    if (a ^ b != true) {
        return 1;
    }

    b = true;
    if (a & b != true) {
        return 1;
    }
    if (a | b != true) {
        return 1;
    }
    if (a ^ b != false) {
        return 1;
    }
    return 0;
}

/// @COMPILE
/// @RUN

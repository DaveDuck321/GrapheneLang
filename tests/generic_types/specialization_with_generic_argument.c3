@require_once "std/arithmetic.c3"
@require_once "std/util.c3"

typedef A : { _ : isize }

typedef [T] Optional : { has_value : bool, value : T }
typedef [T] Optional<T&> : { value : iptr } // Nullptr optimization
typedef Optional<bool> : { value : u8 }

function main: () -> int = {
    if sizeof<Optional<A>>() != 16 {
        return 1;
    }

    if sizeof<Optional<A&>>() != 8 {
        return 2;
    }

    if sizeof<Optional<bool>>() != 1 {
        return 3;
    }

    return 0;
}

/// @COMPILE
/// @RUN

@require_once "std/arithmetic.c3"
@require_once "std/util.c3"

typedef [T] Optional : { has_value : bool, value : T }
typedef [T] Optional<T&> : { value : iptr } // Nullptr optimization
typedef [T] Optional<bool> : { value : u8 }

function main: () -> int = {
    if sizeof<Optional<isize>>() != 16 {
        return 1;
    }

    if sizeof<Optional<int&>>() != 8 {
        return 2;
    }

    if sizeof<Optional<bool>>() != 1 {
        return 3;
    }

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 6
/// Error: generic type 'Optional<bool, T>' with 2 arguments was previously declared with 1 argument:
/// - typedef  [[]T[]] Optional<T&> : { value: iptr }                   (File '*.c3', line 5)
/// - typedef  [[]T[]] Optional<T> : { has_value: bool, value: T }      (File '*.c3', line 4)

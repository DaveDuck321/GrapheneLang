@require_once "std/array.c3"

typedef MyArray : int[3]

@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
}

function main : () -> int = {
    mut array : MyArray;

    let b : int mut& = &mut array[0];
    let c : int mut& = &mut array[1];

    b = 6;  // Sets `array[0]` to 6
    c = 7;  // Sets `array[1]` to 7
    array[2] = 9;
    return array[0] + array[1] + array[2];
}

/// @COMPILE
/// @RUN; EXPECT 22

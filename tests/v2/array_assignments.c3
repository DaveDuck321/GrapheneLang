typedef MyArray : int[3];

@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

function main : () -> int = {
    let array : MyArray;

    let b : int& = &array[0];
    let c : int& = &array[1];

    b = 6;  // Sets `array[0]` to 6
    c = 7;  // Sets `array[1]` to 7
    array[2] = 9;
    return array[0] + array[1] + array[2];
};

/// @COMPILE
/// @RUN; EXPECT 22

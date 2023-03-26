typedef MyArray : int[2];

@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

function main : () -> int = {
    let array : MyArray;

    let b : int& = &array[0];
    let c : int&& = &b;

    b = 6;  // Sets `a` to 6
    c = &array[1]; // Sets `b` to the address of `x`
    b = 5;  // Sets `x` to 5
    return array[0] + array[1] + c;
};

/// @COMPILE
/// @RUN; EXPECT 16

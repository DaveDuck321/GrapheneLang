typedef MyStruct : {a: int, x: int};

@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

function main : () -> int = {
    let struct : MyStruct;

    let b : int& = &struct.a;
    let c : int&& = &b;

    b = 6;  // Sets `a` to 6
    c = &struct.x; // Sets `b` to the address of `x`
    b = 5;  // Sets `x` to 5
    return struct.a + struct.x + c;
};

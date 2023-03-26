@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

function main : () -> int = {
    let x : int = 0;
    let a : int = 0;
    let b : int& = &a;
    let c : int&& = &b;

    b = 6;  // Sets `a` to 6
    c = &x; // Sets `b` to the address of `x`
    b = 5;  // Sets `x` to 5
    return a + x + c;
};

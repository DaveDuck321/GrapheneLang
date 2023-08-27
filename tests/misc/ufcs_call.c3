@operator +: (a: int, b: int) -> int = {
    return __builtin_add(a, b);
}

@operator -: (val : int) -> int = {
    return __builtin_subtract(0, val);
}

function add: (this: int&, that: int) -> int& = {
    this = this + that;
    return &this;
}

function main: () -> int = {
    let x: int = 2;
    return x:add(-1):add(3);
}

/// @COMPILE
/// @RUN; EXPECT 4

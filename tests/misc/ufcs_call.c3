@operator +: (a: int, b: int) -> int = {
    return __builtin_add(a, b);
}

@operator -: (val : int) -> int = {
    return __builtin_subtract(0, val);
}

function add: (this: int mut&, that: int) -> int mut& = {
    this = this + that;
    return &mut this;
}

function main: () -> int = {
    mut x: int = 2;
    return x:add(-1):add(3);
}

/// @COMPILE
/// @RUN; EXPECT 4

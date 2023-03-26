@operator +: (a: int, b: int) -> int = {
    return __builtin_add(a, b);
};

function add: (this: int&, that: int) -> int& = {
    this = this + that;
    return &this;
};

function main: () -> int = {
    let x: int = 2;
    return x:add(-1):add(3);
};

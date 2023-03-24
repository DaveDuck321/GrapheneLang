typedef my_struct : { a: int, b: int };

@operator + : (a: int, b: int) -> int = {
    return __builtin_add(a, b);
};

function main : () -> int = {
    const s : my_struct = { 1, 2 };
    return s.a + s.b;
};

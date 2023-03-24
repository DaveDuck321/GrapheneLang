typedef my_struct : { a: int, b: int };

@constructor : (this: my_struct&) -> void = {
    this.a = 0;
    this.b = 0;
};

@constructor : (this: my_struct&, a_: int) -> void = {
    this.a = a_;
    this.b = 3;
};

@operator + : (a: int, b: int) -> int = {
    return __builtin_add(a, b);
};

function main : () -> int = {
    const s : my_struct = { 2 };
    return s.a + s.b;
};

typedef int_ref : int&;

function d : (var : int_ref&) -> int = { return var; };

function main: () -> int = {
    let a: int = 2;
    let b: int& = &a;
    let c: int_ref& = &b;
    return d(&c);
};

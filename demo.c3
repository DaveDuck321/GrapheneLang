function[T, U] hello: () -> T = {
    let a : T = 1;
    return a;
};

function main: () -> int = {
    hello<i64, int>();
    return 0;
};

foreign puts : (str: string) -> int;
foreign raise : (sig: int) -> int;

function a : () -> int = {
    puts("hello world a");
    return 1;
};

function b : (b: int) -> int = {
    puts("hello world b");
    return 7;
};

function c : () -> int = {
    puts("hello world c");
    return 1;
};

function main: () -> int = {
    a();
    b(c());

    raise(11);

    return a();
};

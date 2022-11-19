foreign puts : (str: string) -> int;

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

    return a();
};

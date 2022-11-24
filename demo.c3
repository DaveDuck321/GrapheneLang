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

@operator +: (a: int, b: int) -> int = {
    return __builtin_add(1, 3);
};

function main: () -> int = {
    a();
    b(c());

    let x: int = 10;
    {
        let s: string = "Hello!";
    }


    let x2: bool = true;
    if (x2) {
        puts("x2 is true");
    }

    let x3: bool = false;
    if (x3) {
        puts("ERROR, x3 is true");
    }


    let y: int = 0;

    // FIXME return 0;
    return 6 + 1 + y;
};

typedef my_i32 : int;
typedef i64_ref : i64&;
typedef my_struct<T> : {a: T, b: my_i32};

foreign puts : (str: string) -> int;
foreign raise : (sig: int) -> int;

function a : () -> int = {
    puts("hello world a");
    return 1;
};

function b : (b: i64) -> i64 = {
    puts("hello world b");
    return 7;
};

function c : () -> int = {
    puts("hello world c");
    return 1;
};

function d : (var : i64_ref) -> int = {
    puts("hello world d");
    return 0;
};

@operator +: (a: int, b: int) -> int = {
    return __builtin_add(a, b);
};

function main: () -> int = {
    a();
    let bb: i64 = b(c());

    d(&bb);

    let x: i64 = 10;
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


    let y: my_i32 = 8;
    let z : my_struct<i32>; // Note: currently UB (struct uninitialized)

    // FIXME return 0;
    return 6 + 1 + z.a;
};

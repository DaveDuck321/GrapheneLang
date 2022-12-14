typedef my_struct[T, T] : {a: T, b: T};

function main: () -> int = {
    // Linker fails without a main().
    return 0;
};

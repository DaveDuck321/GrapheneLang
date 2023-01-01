typedef[T, T] my_struct : {a: T, b: T};

function main: () -> int = {
    // Linker fails without a main().
    return 0;
};

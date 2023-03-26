foreign puts : (str: string) -> int;

function a : () -> int = {
    puts("a");
    return 1;
};

@operator - : (rhs: int) -> int = {
    puts("-");
    return __builtin_add(rhs, 6);
};

function main: () -> int = {
    return -a();
};

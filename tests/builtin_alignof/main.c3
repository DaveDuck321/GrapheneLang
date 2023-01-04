function main: () -> int = {
    let align : isize = __builtin_alignof<i16>();
    return __builtin_narrow<int>(align);
};

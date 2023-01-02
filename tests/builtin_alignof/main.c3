function main: () -> int = {
    let align : u64 = __builtin_alignof<i16>();
    return __builtin_narrow<int>(align);
};

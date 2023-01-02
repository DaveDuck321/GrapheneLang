function main: () -> u64 = {
    let align : u64 = __builtin_alignof<int>();
    return align;
};

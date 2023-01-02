function main: () -> u64 = {
    let a : int = 200;
    let b : u8 = __builtin_narrow<u8>(a);
    let c : u8 = __builtin_narrow<u8>(7);
    return __builtin_add(b, c);
};

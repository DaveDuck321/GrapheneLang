@require_once "std/io.c3";

typedef[Len] string : u8[Len];

function main: () -> int = {
    // let str: string<3> = "abc";

    return puts<14>("abcdefghijklm\n");
};

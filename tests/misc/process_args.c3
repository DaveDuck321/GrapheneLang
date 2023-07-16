@require_once "std/arithmetic.c3"
@require_once "std/syscalls.c3"

function strlen : (cstr: u8[&]) -> isize = {
    let len : isize = 0;
    const null : u8[1]& = "\0";

    while (cstr[len] != null[0]) {
        len = len + 1;
    }

    return len;
}

function main : (argc: int, argv: u8[&][&]) -> int = {
    let i : int = 0;

    while (i < argc) {
        sys_write(/* stdout */ 1, &argv[i], strlen(&argv[i]));
        sys_write(/* stdout */ 1, "\n", 1);
        i = i + 1;
    }

    return argc;
}

/// @COMPILE
/// @RUN a bb ccc; EXPECT 4; EXPECT OUT
/// -
/// a
/// bb
/// ccc

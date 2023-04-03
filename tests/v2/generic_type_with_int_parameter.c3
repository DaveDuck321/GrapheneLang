@require_once "std/io.c3";

typedef[Len] string : u8[Len];

function main: () -> int = {
    // XXX we truncate array types.
    const str: string<3>& = "ab\nXX";

    puts<3>(&str);

    return 0;
};

/// @COMPILE
/// @RUN; EXPECT OUT
/// ab

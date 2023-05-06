@require_once "std/io.c3";

typedef[Len] string : u8[Len];

function main: () -> int = {
    // XXX we truncate array types.
    const str: string<2>& = "abXX";

    puts<2>(&const str);

    return 0;
};

/// @COMPILE
/// @RUN; EXPECT OUT
/// ab

@require_once "std/io.c3"
@require_once "std/format.c3"

typedef MyStruct : {
    a : int,
}

function [IO] write : (io : IO mut&, struct : MyStruct) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    io:write(sv("struct"));
}

function main : () -> int = {
    let struct : MyStruct = {1};
    let var : int = 6999;
    let var2 : u32 = 0x20;
    let string : StringView = sv("other string");
    printf(sv("Hello % world %%%"), struct, var, string, var2);
    print(struct, sv("world"), var, var2);
    return 0;
}


/// @COMPILE
/// @RUN; EXPECT OUT
/// Hello struct world 6999other string20
/// structworld699920

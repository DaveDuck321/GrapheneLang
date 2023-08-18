@require_once "std/io.c3"
@require_once "std/format.c3"

typedef MyStruct : {
    a : int,
}

function [IO] write : (io : IO&, struct : MyStruct) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    return io:write(sv("struct"));
}

function main : () -> int = {
    let struct : MyStruct = {1};
    let var : int = 6999;
    let string : StringView = sv("other string");
    printf(sv("Hello % world %%"), struct, var, string);
    print(struct, sv("world"), var);
    return 0;
}


/// @COMPILE
/// @RUN; EXPECT OUT
/// Hello struct world 6999other string
/// structworld6999

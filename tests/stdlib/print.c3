@require_once "std/io.c3"

typedef MyStruct : {
    a : int,
}

function stringify : (buffer : u8[&], struct : MyStruct) -> StringView = {
    return make_span("struct");
}

function main : () -> int = {
    let struct : MyStruct = {1};
    let var : int = 6999;
    let string : StringView = make_span("other string");
    printf(sv("Hello % world %%"), struct, var, string);
    print(struct, sv("world"), var);
    return 0;
}


/// @COMPILE
/// @RUN; EXPECT OUT
/// Hello struct world 6999other string
/// structworld6999

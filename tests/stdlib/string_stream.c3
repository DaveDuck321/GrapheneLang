@require_once "std/format.c3"
@require_once "std/sstream.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();
    mut ss : StringStream = make<StringStream>(&mut allocator);

    ss:write("");
    ss:write("Hello, world!");
    ss:write("\n");
    ss:write("Some long text to make sure that this reallocates correctly.");
    ss:write("");

    print(ss:as_sv());

    ss:deallocate();

    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// Hello, world!
/// Some long text to make sure that this reallocates correctly.

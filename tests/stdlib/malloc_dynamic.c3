@require_once "std/arithmetic.c3"
@require_once "std/memory.c3"
@require_once "std/wrappers.c3"

typedef MyStruct : {
    a : int,
    b : isize
}


function main : () -> int = {
    let allocator : Allocator = initialize_allocator();
    let span : Span<int> = allocator:allocate_span<int>(30);
    span.data[29] = 42;

    let span_to_deallocate : Span<int> = allocator:allocate_span<int>(6);
    allocator:deallocate_span(span_to_deallocate);

    return span.data[29];
}

/// @COMPILE
/// @RUN; EXPECT 42

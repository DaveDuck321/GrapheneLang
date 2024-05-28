@require_once "std/json.c3"
@require_once "std/string.c3"
@require_once "std/memory.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();
    let input : StringView = sv("[\"\\\"]"); // ["\"]
    let result : Optional<JSON_Node> = json_parse(input, &mut allocator);

    // Should be rejected.
    if result:has_value() {
        return 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN

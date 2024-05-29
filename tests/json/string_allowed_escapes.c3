@require_once "std/json.c3"
@require_once "std/string.c3"
@require_once "std/memory.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();
    let input : StringView = sv("[\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"]"); // ["\"\\\/\b\f\n\r\t"]
    let result : Optional<JSON_Node> = json_parse(input, &mut allocator);

    // Should be accepted.
    if result:has_value() {
        return 0;
    }

    return 1;
}

/// @COMPILE
/// @RUN

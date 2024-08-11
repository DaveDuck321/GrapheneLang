@require_once "std/json.c3"
@require_once "std/string.c3"
@require_once "std/format.c3"
@require_once "std/memory.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();
    let input : StringView = sv("{\"name\":\"program\",\"children\":[{\"name\":\"require_once\",\"children\":[{\"name\":\"ESCAPED_STRING\",\"value\":\"\\\"std/format.c3\\\"\"}]}]}");
    let result : Optional<JSON_Node> = json_parse(input, &mut allocator);

    // Should be accepted.
    if result:has_value() {
        // Note that the order of the keys in the dictionary isn't necessarily
        // preserved.
        print(result:data());
        return 0;
    }

    return 1;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// {"children": [[]{"children": [[]{"name": "ESCAPED_STRING", "value": ""std/format.c3""}[]], "name": "require_once"}[]], "name": "program"}

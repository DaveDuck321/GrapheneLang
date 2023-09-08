@require_once "std/json.c3"
@require_once "std/string.c3"
@require_once "std/format.c3"


function main : () -> int = {
    const input : StringView = sv("{\"name\":\"program\",\"children\":[{\"name\":\"require_once\",\"children\":[{\"name\":\"ESCAPED_STRING\",\"value\":\"\\\"std/format.c3\\\"\"}]}]}");
    let result : Optional<JSON_Node> = json_parse(input);

    // Should be accepted.
    if result:has_value() {
        print(result:data());
        return 0;
    }

    return 1;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// {"name": "program", "children": [[]{"name": "require_once", "children": [[]{"name": "ESCAPED_STRING", "value": "std/format.c3"}[]]}[]]}

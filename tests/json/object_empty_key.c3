@require_once "std/json.c3"
@require_once "std/string.c3"

function main : () -> int = {
    let input : StringView = sv("{\"\":0}");
    let result : Optional<JSON_Node> = json_parse(input);

    // Should be accepted.
    if result:has_value() {
        return 0;
    }

    return 1;
}

/// @COMPILE
/// @RUN

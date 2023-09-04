@require_once "std/json.c3"
@require_once "std/string.c3"

function main : () -> int = {
    const input : StringView = sv("[\"\"],");
    let result : Optional<JSON_Node> = json_parse(input);

    // Should be rejected.
    if result:has_value() {
        return 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN

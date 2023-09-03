@require_once "std/json.c3"
@require_once "std/string.c3"
@require_once "std/format.c3"

function main : () -> int = {
    let node_opt : Optional<JSON_Node> = json_parse(sv("[\"hello\", 5, -1, {\"hi\": \"bye\"}]"));

    print(node_opt:has_value());

    let node : JSON_Node = node_opt:data();

    printf(sv("type %, value %, children %"), node.type, node.value:has_value(),
           node.children:has_value());

    return 9;
}

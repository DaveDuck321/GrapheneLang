@require_once "std/arithmetic.c3"
@require_once "std/string.c3"
@require_once "std/memory.c3"
@require_once "std/vector.c3"
@require_once "std/wrappers.c3"
@require_once "std/iterators.c3"

typedef Cursor : {
    data      : StringView,
    position  : isize, // TODO should this be an iterator?
    allocator : Allocator,  // FIXME use global allocator.
}

typedef JSON_Node : {
    type     : int, // 0: null, 1: true, 2: false, 3: string, 4: number,
                    // 5: array, 6: object
    value    : Optional<StringView>,
    children : Optional<Vector<JSON_Node>>,
}

// TODO something like new<JSON_Node> would be nice, but we don't support
// function specializations.
function new_json_node : (node_type : int) -> JSON_Node = {
    return {
        node_type,
        make_optional<StringView>(),
        make_optional<Vector<JSON_Node>>(),
    };
}

function char : (literal : u8[1]&) -> u8 = {
    return literal[0];
}

function is_eof : (cursor : Cursor&) -> bool = {
    return cursor.position >= cursor.data.length;
}

function peak : (cursor : Cursor&) -> Optional<u8> = {
    if !cursor:is_eof() {
        return make_optional(cursor.data.data[cursor.position]);
    }

    return make_optional<u8>();
}

function next : (cursor : Cursor&) -> Optional<u8> = {
    // TODO this should be const (can't use with has_value() right now).
    let token : Optional<u8> = cursor:peak();

    if token:has_value() {
        cursor.position += 1;
    }

    return token;
}

function is_json_whitespace : (token : u8) -> bool = {
    // JSON defines whitespace as one of { space, linefeed, carriage return,
    // horizontal tab }. If we use string literals instead of code points, then
    // our code doesn't care about the underlying encoding :).
    return token == char(" ") or token == char("\n")
           or token == char("\r") or token == char("\t");
}

function eat : (cursor : Cursor&, token : u8) -> bool = {
    // TODO this should be const (can't use with has_value() right now).
    let next_char : Optional<u8> = cursor:peak();

    let next_char_data : u8 = next_char:data();
    if next_char:has_value() and next_char_data == token {
        cursor:next();
        return true;
    }

    return false;
}

function eat : (cursor : Cursor&, string : StringView) -> bool = {
    for token in string:get_iter() {
        if !cursor:eat(token) {
            return false;
        }
    }

    return true;
}

function eat_whitespace : (cursor: Cursor&) -> void = {
    while !cursor:is_eof() {
        let next_char : Optional<u8> = cursor:peak();
        runtime_assert(next_char:has_value());

        if !is_json_whitespace(next_char:data()) {
            return;
        }

        // Eat the current character.
        cursor:next();
    }
}

// TODO return ErrorOr<JSON_Node>.
function parse_object : (cursor : Cursor&) -> Optional<JSON_Node> = {
    if !cursor:eat(char("{")) {
        return make_optional<JSON_Node>();
    }

    let node : JSON_Node = new_json_node(/* object */ 6);

    cursor:eat_whitespace();

    if cursor:eat(char("}")) {
        // Empty object.
        return make_optional(node);
    }

    // Initialize children vector.
    node.children:store(cursor.allocator:make_vector<JSON_Node>());

    while true {
        cursor:eat_whitespace();

        let string_node : Optional<JSON_Node> = cursor:parse_string();

        if !string_node:has_value() {
            node.children:data():deallocate();
            return make_optional<JSON_Node>();
        }

        node.children:data():push_back(string_node:data());

        cursor:eat_whitespace();

        if !cursor:eat(char(":")) {
            node.children:data():deallocate();
            return make_optional<JSON_Node>();
        }

        let value_node : Optional<JSON_Node> = cursor:parse_element();

        if !value_node:has_value() {
            node.children:data():deallocate();
            return make_optional<JSON_Node>();
        }

        node.children:data():push_back(value_node:data());

        if cursor:eat(char("}")) {
            // We're done.
            return make_optional(node);
        }

        if !cursor:eat(char(",")) {
            node.children:data():deallocate();
            return make_optional<JSON_Node>();
        }
    }

    // Unreachable.
    return make_optional(node);
}

function parse_array : (cursor : Cursor&) -> Optional<JSON_Node> = {
    if !cursor:eat(char("[")) {
        return make_optional<JSON_Node>();
    }

    let node : JSON_Node = new_json_node(/* array */ 5);

    cursor:eat_whitespace();

    if cursor:eat(char("]")) {
        // Empty array.
        return make_optional(node);
    }

    // Initialize children vector.
    node.children:store(cursor.allocator:make_vector<JSON_Node>());

    while true {
        let value_node : Optional<JSON_Node> = cursor:parse_element();

        if !value_node:has_value() {
            return make_optional<JSON_Node>();
        }

        node.children:data():push_back(value_node:data());

        if cursor:eat(char("]")) {
            // We're done.
            return make_optional(node);
        }

        if !cursor:eat(char(",")) {
            node.children:data():deallocate();
            return make_optional<JSON_Node>();
        }
    }

    // Unreachable.
    return make_optional<JSON_Node>();
}

function parse_value : (cursor : Cursor&) -> Optional<JSON_Node> = {
    let optional_next_token : Optional<u8> = cursor:peak();
    if !optional_next_token:has_value() {
        return make_optional<JSON_Node>();
    }

    const next_token : u8 = optional_next_token:data();

    if next_token == char("\"") {
        return cursor:parse_string();
    }

    if next_token == char("{") {
        return cursor:parse_object();
    }

    if next_token == char("[") {
        return cursor:parse_array();
    }

    // Numbers don't have a prefix character, so just try and parse them.
    let node : Optional<JSON_Node> = cursor:parse_number();
    if node:has_value() {
        return node;
    }

    if cursor:eat(sv("null")) {
        return make_optional(new_json_node(/* null */ 0));
    }

    if cursor:eat(sv("true")) {
        return make_optional(new_json_node(/* true */ 1));
    }

    if cursor:eat(sv("false")) {
        return make_optional(new_json_node(/* false */ 2));
    }

    return make_optional<JSON_Node>();
}

function parse_element : (cursor : Cursor&) -> Optional<JSON_Node> = {
    cursor:eat_whitespace();

    const value_node : Optional<JSON_Node> = cursor:parse_value();

    cursor:eat_whitespace();

    return value_node;
}

function parse_string : (cursor : Cursor&) -> Optional<JSON_Node> = {
    if !cursor:eat(char("\"")) {
        return make_optional<JSON_Node>();
    }

    const start_pos : isize = cursor.position;

    while !cursor:is_eof() {
        if cursor:eat(char("\"")) {
            const end_pos : isize = cursor.position - 1;

            let node : JSON_Node = new_json_node(/* string */ 3);
            node.value = make_optional(
                cursor.data:slice_between(start_pos, end_pos)
            );

            return make_optional(node);
        }

        // TODO escape characters.

        cursor:next();
    }

    return make_optional<JSON_Node>();
}

function is_digit_helper : (token : Optional<u8>) -> bool = {
    return token:has_value() and ascii_is_digit(token:data());
}

function parse_number : (cursor : Cursor&) -> Optional<JSON_Node> = {
    const start_pos : isize = cursor.position;
    let is_valid : bool = false;

    // This is optional, so discard the return value.
    cursor:eat(char("-"));

    if !cursor:eat(char("0")) {
        while is_digit_helper(cursor:peak()) {
            is_valid = true;
            cursor:next();
        }
    }

    // TODO parse fraction.
    // TODO parse exponent.

    const end_pos : isize = cursor.position;

    if !is_valid {
        return make_optional<JSON_Node>();
    }

    let node : JSON_Node = new_json_node(/* string */ 4);
    node.value = make_optional(cursor.data:slice_between(start_pos, end_pos));

    return make_optional(node);
}

// TODO this and write below should be the only public functions.
function json_parse : (source : StringView) -> Optional<JSON_Node> = {
    // FIXME memory safety, memory leaks.
    let cursor : Cursor = {
        source, 0, initialize_allocator()
    };

    let result : Optional<JSON_Node> = cursor:parse_element();

    if !result:has_value() {
        return result;
    }

    // Only succeed if we have consumed the entire input.
    // e.g. '[1, 2], 2' would parse up to the end of the array, but the
    // remaining input makes this invalid json.
    cursor:eat_whitespace();
    if cursor:is_eof() {
        return result;
    }

    return make_optional<JSON_Node>();
}

function [IO] write : (io : IO&, node : JSON_Node) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    // null
    if node.type == 0 {
        return io:write(sv("null"));
    }

    // true
    if node.type == 1 {
        return io:write(sv("true"));
    }

    // false
    if node.type == 2 {
        return io:write(sv("false"));
    }

    // string
    if node.type == 3 {
        // Allowed to fail.
        if node.value:has_value() {
            let string : StringView = node.value:data();
            let buffer : Span<u8> = io:request_buffer(string.length + 2);

            buffer.data[0] = char("\"");
            buffer.data[buffer.length - 1] = char("\"");
            for i in range(string.length) {
                buffer.data[i + 1] = string.data[i];
            }

            io:consume_length(buffer.length);
            return {&buffer.data, buffer.length};
        }
    }

    // number (which we can just echo back!)
    if node.type == 4 {
        // Allowed to fail.
        if node.value:has_value() {
            return io:write(node.value:data());
        }
    }

    // array
    if node.type == 5 {
        // FIXME we should be using a buffer here.
        io:write(sv("["));
        if node.children:has_value() {
            let iter : VectorIter<JSON_Node> = node.children:data():get_iter();
            for element in &iter {
                io:write(element);
                if iter:has_next() {
                    io:write(", ");
                }
            }
        }
        return io:write(sv("]"));
    }

    // object
    if node.type == 6 {
        // FIXME we should be using a buffer here.
        io:write(sv("{"));
        if node.children:has_value() {
            let iter : VectorIter<JSON_Node> = node.children:data():get_iter();
            let flag : bool = true;

            // XXX we have to borrow the iterator, otherwise it is not updated
            // in place by the implicit call to get_next().
            for element in &iter {
                io:write(element);

                if flag {
                    io:write(sv(": "));
                } else {
                    if iter:has_next() {
                        io:write(sv(", "));
                    }
                }
                flag ^= true;
            }
        }
        return io:write(sv("}"));
    }

    return io:write(sv("<invalid node>"));
}

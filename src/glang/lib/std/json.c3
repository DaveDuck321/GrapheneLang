@require_once "std/arithmetic.c3"
@require_once "std/iterators.c3"
@require_once "std/logical.c3"
@require_once "std/map.c3"
@require_once "std/memory.c3"
@require_once "std/string.c3"
@require_once "std/vector.c3"
@require_once "std/wrappers.c3"

typedef StringViewIter : ContiguousContainerIter<View<u8>, u8&>
typedef VectorIter<JSON_Node> : ContiguousContainerIter<Vector<JSON_Node>&, JSON_Node mut&>

typedef Cursor : {
    iter      : StringViewIter,
    allocator : Allocator mut&,  // FIXME use global allocator.
    // TODO when we get RAII working, add a max_depth field to prevent crashes
    // when the input is too nested.
}

typedef JSON_Node : {
    type     : int, // 0: null, 1: true, 2: false, 3: string, 4: number,
                    // 5: array, 6: object
    value    : Optional<StringView>,
    children : Vector<JSON_Node>,
    map      : Map<StringView, JSON_Node>,
}

function make<JSON_Node> : (node_type : int, allocator : Allocator mut&) -> JSON_Node = {
    return {
        node_type,
        make<Optional<StringView>>(),
        make<Vector<JSON_Node>>(&mut allocator),
        make<Map<StringView, JSON_Node>>(&mut allocator),
    };
}

function deallocate : (self : JSON_Node mut&) -> void = {
    // FIXME can't iterate with get_iter(); it returns an immutable reference
    // for some reason.
    for i in range(self.children:length()) {
        self.children:get(i):deallocate();
    }
    self.children:deallocate();

    self.map:clear();
}

function has_next : (cursor : Cursor mut&) -> bool = {
    return cursor.iter:has_next();
}

function peek : (cursor : Cursor mut&) -> Optional<u8> = {
    if cursor.iter:has_next() {
        return make<Optional<u8>>(cursor.iter:peek());
    }
    return make<Optional<u8>>();
}

function next : (cursor : Cursor mut&) -> Optional<u8> = {
    if cursor.iter:has_next() {
        return make<Optional<u8>>(cursor.iter:__builtin_get_next());
    }

    return make<Optional<u8>>();
}

function position : (cursor : Cursor&) -> isize = {
    // Looking into the iterator like this is sorta cursed
    return cursor.iter.current_index;
}

function slice_between : (cursor: Cursor&, start_offset : iptr, end_offset : isize) -> StringView = {
    // Looking into the iterator like this is sorta cursed
    let slice_view : View<u8> = cursor.iter.container:slice_between(start_offset, end_offset);
    return { slice_view };
}

function is_json_whitespace : (token : u8) -> bool = {
    // JSON defines whitespace as one of { space, linefeed, carriage return,
    // horizontal tab }. If we use string literals instead of code points, then
    // our code doesn't care about the underlying encoding :).
    return token == char(" ") or token == char("\n")
           or token == char("\r") or token == char("\t");
}

function eat : (cursor : Cursor mut&, token : u8) -> bool = {
    // TODO this should be let (can't use with has_value() right now).
    let next_char : Optional<u8> = cursor:peek();

    if next_char:has_value() and next_char:data() == token {
        cursor:next();
        return true;
    }

    return false;
}

function eat : (cursor : Cursor mut&, string : StringView) -> bool = {
    for token in string:get_iter() {
        if !cursor:eat(token) {
            return false;
        }
    }

    return true;
}

function eat_whitespace : (cursor: Cursor mut&) -> void = {
    while cursor:has_next() {
        let next_char : Optional<u8> = cursor:peek();
        runtime_assert(next_char:has_value());

        if !is_json_whitespace(next_char:data()) {
            return;
        }

        // Eat the current character.
        cursor:next();
    }
}

// TODO return ErrorOr<JSON_Node>.
function parse_object : (cursor : Cursor mut&) -> Optional<JSON_Node> = {
    if !cursor:eat(char("{")) {
        return make<Optional<JSON_Node>>();
    }

    mut node : JSON_Node = make<JSON_Node>(/* object */ 6, &mut cursor.allocator);

    cursor:eat_whitespace();

    if cursor:eat(char("}")) {
        // Empty object.
        return make<Optional<JSON_Node>>(node);
    }

    while true {
        cursor:eat_whitespace();

        let string_node : Optional<JSON_Node> = cursor:parse_string();

        if !string_node:has_value() {
            node.map:clear();
            return make<Optional<JSON_Node>>();
        }

        runtime_assert(string_node:data().value:has_value());
        let string_view : StringView = string_node:data().value:data();

        cursor:eat_whitespace();

        if !cursor:eat(char(":")) {
            node.map:clear();
            return make<Optional<JSON_Node>>();
        }

        let value_node : Optional<JSON_Node> = cursor:parse_element();

        if !value_node:has_value() {
            node.map:clear();
            return make<Optional<JSON_Node>>();
        }

        node.map:insert(string_view, value_node:data());

        if cursor:eat(char("}")) {
            // We're done.
            return make<Optional<JSON_Node>>(node);
        }

        if !cursor:eat(char(",")) {
            node.map:clear();
            return make<Optional<JSON_Node>>();
        }
    }

    // Unreachable.
    runtime_assert(false);
    return make<Optional<JSON_Node>>(node);
}

function parse_array : (cursor : Cursor mut&) -> Optional<JSON_Node> = {
    if !cursor:eat(char("[")) {
        return make<Optional<JSON_Node>>();
    }

    mut node : JSON_Node = make<JSON_Node>(/* array */ 5, &mut cursor.allocator);

    cursor:eat_whitespace();

    if cursor:eat(char("]")) {
        // Empty array.
        return make<Optional<JSON_Node>>(node);
    }

    while true {
        let value_node : Optional<JSON_Node> = cursor:parse_element();

        if !value_node:has_value() {
            return make<Optional<JSON_Node>>();
        }

        node.children:push_back(value_node:data());

        if cursor:eat(char("]")) {
            // We're done.
            return make<Optional<JSON_Node>>(node);
        }

        if !cursor:eat(char(",")) {
            node.children:deallocate();
            return make<Optional<JSON_Node>>();
        }
    }

    // Unreachable.
    return make<Optional<JSON_Node>>();
}

function parse_value : (cursor : Cursor mut&) -> Optional<JSON_Node> = {
    let optional_next_token : Optional<u8> = cursor:peek();
    if !optional_next_token:has_value() {
        return make<Optional<JSON_Node>>();
    }

    let next_token : u8 = optional_next_token:data();

    if next_token == char("\"") {
        return cursor:parse_string();
    }

    if next_token == char("{") {
        return cursor:parse_object();
    }

    if next_token == char("[") {
        return cursor:parse_array();
    }

    if cursor:eat(sv("null")) {
        return make<Optional<JSON_Node>>(make<JSON_Node>(/* null */ 0, &mut cursor.allocator));
    }

    if cursor:eat(sv("true")) {
        return make<Optional<JSON_Node>>(make<JSON_Node>(/* true */ 1, &mut cursor.allocator));
    }

    if cursor:eat(sv("false")) {
        return make<Optional<JSON_Node>>(make<JSON_Node>(/* false */ 2, &mut cursor.allocator));
    }

    // Numbers don't have a prefix character, so just try and parse them.
    return cursor:parse_number();
}

function parse_element : (cursor : Cursor mut&) -> Optional<JSON_Node> = {
    cursor:eat_whitespace();

    let value_node : Optional<JSON_Node> = cursor:parse_value();

    cursor:eat_whitespace();

    return value_node;
}

function is_control_character : (char : u8) -> bool = {
    // All Unicode characters may be placed within the quotation marks, except
    // for the characters that MUST be escaped: quotation mark, reverse
    // solidus, and the control characters (U+0000 through U+001F). [RFC 8259]
    // TODO maybe we should be able to compare unsigned values... char <= 0x1f
    // is so much cleaner, although they both compile to the same code with -O3.
    return (char & ~0x1f) == 0x0;
}

function parse_string : (cursor : Cursor mut&) -> Optional<JSON_Node> = {
    if !cursor:eat(char("\"")) {
        return make<Optional<JSON_Node>>();
    }

    mut in_escape_seq : bool = false;
    // FIXME stop leaking the vector.
    mut string : Vector<u8> = make<Vector<u8>>(&mut cursor.allocator);

    while cursor:has_next() {
        if !in_escape_seq and cursor:eat(char("\"")) {
            mut node : JSON_Node = make<JSON_Node>(/* string */ 3, &mut cursor.allocator);

            node.value = make<Optional<StringView>>(
                make<StringView>(string:view())
            );

            return make<Optional<JSON_Node>>(node);
        }

        let char_opt : Optional<u8> = cursor:next();
        runtime_assert(char_opt:has_value());
        let char : u8 = char_opt:data();

        // Reject strings with control characters.
        if is_control_character(char) {
            return make<Optional<JSON_Node>>();
        }

        if in_escape_seq {
            in_escape_seq = false;

            if char == char("\"") {
                string:push_back(char("\""));
            }
            if char == char("\\") {
                string:push_back(char("\\"));
            }
            if char == char("/") {
                string:push_back(char("/"));
            }
            if char == char("b") {
                string:push_back(char("\b"));
            }
            if char == char("f") {
                string:push_back(char("\f"));
            }
            if char == char("n") {
                string:push_back(char("\n"));
            }
            if char == char("r") {
                string:push_back(char("\r"));
            }
            if char == char("t") {
                string:push_back(char("\t"));
            }
        }
        else {
            if char == char("\\") {
                in_escape_seq = true;
            }
            else {
                string:push_back(char);
            }
        }
    }

    return make<Optional<JSON_Node>>();
}

function is_digit_helper : (token : Optional<u8>) -> bool = {
    return token:has_value() and ascii_is_digit(token:data());
}

function parse_number : (cursor : Cursor mut&) -> Optional<JSON_Node> = {
    let start_pos : isize = cursor:position();
    mut is_valid : bool = false;

    // This is optional, so discard the return value.
    cursor:eat(char("-"));

    if !cursor:eat(char("0")) {
        while is_digit_helper(cursor:peek()) {
            is_valid = true;
            cursor:next();
        }
    }
    else {
        is_valid = true;
    }

    // TODO parse fraction.
    // TODO parse exponent.

    let end_pos : isize = cursor:position();

    if !is_valid {
        return make<Optional<JSON_Node>>();
    }

    mut node : JSON_Node = make<JSON_Node>(/* string */ 4, &mut cursor.allocator);
    node.value = make<Optional<StringView>>(cursor:slice_between(start_pos, end_pos));

    return make<Optional<JSON_Node>>(node);
}

// TODO this and write below should be the only public functions.
function json_parse : (source : StringView, allocator : Allocator mut&) -> Optional<JSON_Node> = {
    // FIXME memory safety, memory leaks.
    mut cursor : Cursor = { source:get_iter(), &mut allocator };

    let result : Optional<JSON_Node> = cursor:parse_element();

    if !result:has_value() {
        return result;
    }

    // Only succeed if we have consumed the entire input.
    // e.g. '[1, 2], 2' would parse up to the end of the array, but the
    // remaining input makes this invalid json.
    cursor:eat_whitespace();
    if !cursor:has_next() {
        return result;
    }

    return make<Optional<JSON_Node>>();
}

function [IO] write : (io : IO mut&, node : JSON_Node) -> TypeIf<void, IsStreamableTextIO<IO>> = {
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

            io:write_char(char("\""));

            for char in string:get_iter() {
                if char == char("\"") {
                    io:write("\\\"");
                    continue;
                }
                if char == char("\\") {
                    io:write("\\\\");
                    continue;
                }
                if char == char("/") {
                    io:write("\\/");
                    continue;
                }
                if char == char("\b") {
                    io:write("\\b");
                    continue;
                }
                if char == char("\f") {
                    io:write("\\f");
                    continue;
                }
                if char == char("\n") {
                    io:write("\\n");
                    continue;
                }
                if char == char("\r") {
                    io:write("\\r");
                    continue;
                }
                if char == char("\t") {
                    io:write("\\t");
                    continue;
                }

                io:write_char(char);
            }

            io:write_char(char("\""));
            return;
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

        mut iter : VectorIter<JSON_Node> = node.children:get_iter();
        for element in &mut iter {
            io:write(element);
            if iter:has_next() {
                io:write(sv(", "));
            }
        }

        return io:write(sv("]"));
    }

    // object
    if node.type == 6 {
        // FIXME we should be using a buffer here.
        io:write(sv("{"));

        mut iter : MapIter<StringView, JSON_Node> = node.map:get_citer();

        // XXX we have to borrow the iterator, otherwise it is not updated
        // in place by the implicit call to get_next().
        for element in &mut iter {
            // FIXME we need to escape the strings we print here.
            io:writef(sv("\"%\": %"), element.key, element.value);
            if iter:has_next() {
                io:write(sv(", "));
            }
        }

        return io:write(sv("}"));
    }

    io:write(sv("<invalid node>"));
}

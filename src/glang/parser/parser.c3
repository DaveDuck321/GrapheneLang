@require_once "file_info.c3"
@require_once "syntax_tree.c3"

@require_once "std/algorithms.c3"
@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/iterators.c3"
@require_once "std/memory.c3"

typedef Cursor : {
    program : StringView,
    position : isize,
    allocator: Allocator mut&, // TODO: this should be somewhere else?
}

function get_sv : (cursor : Cursor&, slice : ProgramSlice) -> StringView = {
    return { slice_between(&cursor.program.buffer, slice.start_offset, slice.end_offset) };
}

function slice : (cursor : Cursor&, length : isize) -> ProgramSlice = {
    return {cursor.position, cursor.position + length};
}

function ascii_is_text_token_start_char : (char : u8) -> bool = {
    return ascii_is_alpha(char) or char == "_"[0];
}

function ascii_is_text_token_char : (char : u8) -> bool = {
    return ascii_is_alphanumeric(char) or char == "_"[0];
}

function get_char : (cursor : Cursor&) -> u8 = {
    return cursor:get_char_at_offset(0);
}

function get_char_at_offset : (cursor : Cursor&, offset : isize) -> u8 = {
    if cursor.position + offset >= cursor.program:length() {
        return "\0"[0];
    }
    // TODO: allow direct indexing into a stringview
    return cursor.program.buffer.data[cursor.position + offset];
}

function meta : (start : isize, end : isize) -> Meta = {
    return {start, end};
}

function [@Len] is_next : (cursor : Cursor&, token : u8[@Len]&) -> bool = {
    return cursor:is_next(sv(&token));
}

function is_next : (cursor : Cursor&, token : StringView) -> bool = {
    for i in range(token:length()) {
        if cursor:get_char_at_offset(i) != token.buffer.data[i] {
            return false;
        }
    }

    if ascii_is_text_token_start_char(token.buffer.data[0]) {
        // Avoid matching a substring in a larger text token
        return !ascii_is_text_token_char(cursor:get_char_at_offset(token:length()));
    }
    return true;
}

function [@Len] try_skip_next_and_surrounding_white_space : (cursor : Cursor mut&, token : u8[@Len]&) -> bool = {
    return cursor:try_skip_next_and_surrounding_white_space(sv(&token));
}

function try_skip_next_and_surrounding_white_space : (cursor : Cursor mut&, token : StringView) -> bool = {
    cursor:try_skip_white_space();
    let result : bool = cursor:try_skip_next(token);
    cursor:try_skip_white_space();
    return result;
}

function try_skip_next_operator : (cursor : Cursor mut&, operator: StringView) -> bool = {
    // The small lookahead exists so we don't capture the operator part of an assignment
    cursor:try_skip_white_space();
    if cursor:get_char_at_offset(operator:length()) == "="[0] {
        return false;
    }
    if cursor:is_next("<<=") or cursor:is_next(">>=") or cursor:is_next("**=") {
        return false;
    }
    return cursor:try_skip_next(operator);
}

function [@Len] try_skip_next : (cursor : Cursor mut&, string : u8[@Len]&) -> bool = {
    return cursor:try_skip_next(sv(&string));
}

function try_skip_next : (cursor : Cursor mut&, string : StringView) -> bool = {
    if !cursor:is_next(string) {
        return false;
    }
    cursor.position += string:length();
    return true;
}

function skip_ws_return_offset : (cursor : Cursor mut&) -> isize = {
    let start_offset : isize = cursor.position;
    cursor:try_skip_white_space();
    return start_offset;
}

function [IO] write_syntax_error_location : (io : IO mut&, cursor : Cursor mut&, position: isize) -> void = {
    let lookup_table : PositionLookupTable = generate_position_lookup_table(cursor.program, &mut cursor.allocator);
    let loc : FilePosition = lookup_table:lookup_position(position);

    io:writef(
        sv("{\"line\":%, \"column\":%}"),
        loc.line,
        loc.column,
    );
}

function [Args...] syntax_error : (cursor : Cursor mut&, format_args : Args...) -> void = {
    syntax_error_at_position(&mut cursor, cursor.position, format_args...);
}

// TODO: ugly error if we forget the generic definition
function [Args...] syntax_error_at_position : (cursor : Cursor mut&, position : isize, format_args : Args...) -> void = {
    mut io : BufferedFile = get_stdout_io();
    io:write("{");
    io:write("\"success\":false");
    io:write(",\"location\":");
    io:write_syntax_error_location(&mut cursor, position);
    io:write(",\"message\":");
    io:write("\"");
    io:writef(format_args...);
    io:write("\"");
    io:write("}");
    io:close();
    sys_exit(1);
}

function try_skip_comment : (cursor : Cursor mut&) -> bool = {
    // Single line comment
    if cursor:is_next("//") {
        while !cursor:is_next("\n") and !cursor:is_next("\0") {
            cursor.position += 1;
        }
        cursor.position += 1;
        return true;
    }

    // Multi-line comment
    if cursor:is_next("/*") {
        while !cursor:is_next("*/") and !cursor:is_next("\0") {
            cursor.position += 1;
        }
        cursor.position += 2;
        return true;
    }

    // Nothing to skip
    return false;
}

function try_skip_ascii_white_space : (cursor : Cursor mut&) -> bool = {
    mut has_skipped : bool = false;
    while ascii_is_white_space(cursor:get_char()) {
        cursor.position += 1;
        has_skipped = true;
    }
    return has_skipped;
}

function try_skip_white_space : (cursor : Cursor mut&) -> bool = {
    // TODO: `do while` or similar
    mut has_skipped_this_loop : bool = true;
    mut has_skipped_any : bool = false;

    while has_skipped_this_loop {
        has_skipped_this_loop = false;
        has_skipped_this_loop |= cursor:try_skip_ascii_white_space();
        has_skipped_this_loop |= cursor:try_skip_comment();
        has_skipped_any |= has_skipped_this_loop;
    }
    return has_skipped_any;
}

function [@Len] expect_next : (cursor : Cursor mut&, next : u8[@Len]&) -> void = {
    if !cursor:try_skip_next(&next) {
        syntax_error(&mut cursor, sv("expected '%'"), sv(&next));
    }
}

function [@Len] expect_next_skipping_ws : (cursor : Cursor mut&, next : u8[@Len]&) -> void = {
    cursor:try_skip_white_space();
    cursor:expect_next(&next);
    cursor:try_skip_white_space();
}

function is_end_of_token : (cursor : Cursor mut&) -> bool = {
    if cursor:try_skip_white_space() {
        // White space can always end a token
        return true;
    }

    if !ascii_is_text_token_char(cursor:get_char()) {
        return true;
    }

    return false;
}

function get_next_hex_token : (cursor : Cursor mut&) -> ProgramSlice = {
    runtime_assert(cursor:is_next("0x"));

    mut token : ProgramSlice = cursor:slice(2);
    cursor.position += 2;

    while !cursor:is_end_of_token() {
        let next_char : u8 = ascii_to_lower(cursor:get_char());
        if ascii_is_digit(next_char) |
          (next_char == "a"[0]) |
          (next_char == "b"[0]) |
          (next_char == "c"[0]) |
          (next_char == "d"[0]) |
          (next_char == "e"[0]) |
          (next_char == "f"[0]) {
            cursor.position += 1;
            token.end_offset += 1;
        } else {
            // Otherwise its a syntax error
            syntax_error(&mut cursor, sv("illegal digit '%' in hex constant"), decode_utf8(next_char));
        }
    }
    if token.end_offset - token.start_offset == 2 {
        syntax_error(&mut cursor, sv("expected hex constant continuation"));
    }

    return token;
}

function get_next_bool_token : (cursor : Cursor mut&) -> ProgramSlice = {
    let true_str : StringView = sv("true");
    let false_str : StringView = sv("false");

    runtime_assert(cursor:is_next(true_str) or cursor:is_next(false_str));

    let start_offset : isize = cursor.position;
    if cursor:try_skip_next_and_surrounding_white_space(true_str) {
        return {start_offset, start_offset + true_str:length()};
    } else {
        runtime_assert(cursor:try_skip_next_and_surrounding_white_space(false_str));
        return {start_offset, start_offset + false_str:length()};
    }
}

// TODO: merge constant and compile time constant
typedef NumericToken : {
    data : ProgramSlice,
    type : int, // 1: float, 2: int, 5: hex (matching `Constant`)
}

function get_next_numeric_token : (cursor : Cursor mut&) -> NumericToken = {
    // TODO: binary/ octal
    if cursor:is_next("0x") {
        return {cursor:get_next_hex_token(), 5 /* Hex */};
    }

    // Parse floating point and integer constants
    mut has_seen_decimal_point : bool = false;
    mut has_seen_exponent : bool = false;

    mut token : ProgramSlice = cursor:slice(0);
    if cursor:try_skip_next("-") {
        token.end_offset += 1;
    }

    while !cursor:is_end_of_token() {
        let next_char : u8 = cursor:get_char();
        if ascii_is_digit(next_char) {
            cursor.position += 1;
            token.end_offset += 1;

            if cursor:get_char() == char(".") {
                if has_seen_decimal_point {
                    syntax_error(&mut cursor, sv("cannot have two decimal points in a floating point constant"));
                }
                if has_seen_exponent {
                    syntax_error(&mut cursor, sv("cannot have decimal place in floating point exponent"));
                }
                has_seen_decimal_point = true;

                cursor.position += 1;
                token.end_offset += 1;
            }

            if ascii_to_upper(cursor:get_char()) == char("E") {
                if has_seen_exponent {
                    syntax_error(&mut cursor, sv("cannot have two exponents in a floating point constant"));
                }
                has_seen_exponent = true;

                cursor.position += 1;
                token.end_offset += 1;

                if cursor:get_char() == char("-") or cursor:get_char() == char("+") {
                    cursor.position += 1;
                    token.end_offset += 1;
                }

                if !ascii_is_digit(cursor:get_char()) {
                    syntax_error(&mut cursor, sv("illegal digit '%' in floating constant"), decode_utf8(next_char));
                }
            }
        } else {
            // Otherwise its a syntax error
            syntax_error(&mut cursor, sv("illegal digit '%' in decimal constant"), decode_utf8(next_char));
        }
    }
    if token.end_offset - token.start_offset == 0 {
        syntax_error(&mut cursor, sv("expected decimal constant"));
    }

    if has_seen_decimal_point or has_seen_exponent {
        return {token, 1 /* Floating point */};
    } else {
        return {token, 2 /* Int */};
    }
}

function get_next_string_literal_token : (cursor : Cursor mut&) -> ProgramSlice = {
    runtime_assert(cursor:try_skip_next("\""));
    let start_offset : isize = cursor.position;

    while !cursor:try_skip_next("\"") {
        if cursor:is_next("\0") {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated string literal"));
        }

        if cursor:is_next("\\") {
            // Skip the character after an escape sequence
            // TODO: allow numeric escape sequences eg \789
            cursor.position += 1;
        }
        cursor.position += 1;
    }
    // The string has been terminated
    let result : ProgramSlice = {start_offset, cursor.position-1};
    cursor:try_skip_white_space();
    return result;
}

function get_next_operator : (cursor : Cursor mut&) -> ProgramSlice = {
    let start_position : isize = cursor.position;

    if cursor:try_skip_next_and_surrounding_white_space("==")
        or cursor:try_skip_next_and_surrounding_white_space("<=")
        or cursor:try_skip_next_and_surrounding_white_space(">=")
        or cursor:try_skip_next_and_surrounding_white_space("!=")
        or cursor:try_skip_next_and_surrounding_white_space("<<")
        or cursor:try_skip_next_and_surrounding_white_space(">>")
        or cursor:try_skip_next_and_surrounding_white_space("**")
    {
        return {start_position, start_position + 2};
    }

    if cursor:try_skip_next_and_surrounding_white_space("<")
        or cursor:try_skip_next_and_surrounding_white_space(">")
        or cursor:try_skip_next_and_surrounding_white_space("!")
        or cursor:try_skip_next_and_surrounding_white_space("~")
        or cursor:try_skip_next_and_surrounding_white_space("+")
        or cursor:try_skip_next_and_surrounding_white_space("-")
        or cursor:try_skip_next_and_surrounding_white_space("*")
        or cursor:try_skip_next_and_surrounding_white_space("/")
        or cursor:try_skip_next_and_surrounding_white_space("%")
        or cursor:try_skip_next_and_surrounding_white_space("|")
        or cursor:try_skip_next_and_surrounding_white_space("^")
        or cursor:try_skip_next_and_surrounding_white_space("@")
        or cursor:try_skip_next_and_surrounding_white_space("&")
    {
        return {start_position, start_position + 1};
    }

    syntax_error(&mut cursor, sv("illegal operator '%'"), decode_utf8(cursor:get_char()));
    return {0, 0};
}

function get_next_assignment : (cursor : Cursor mut&) -> ProgramSlice = {
    let start_position : isize = cursor.position;

    if cursor:try_skip_next_and_surrounding_white_space("=") {
        return {start_position, start_position + 1};
    }

    if cursor:try_skip_next_and_surrounding_white_space("<<=")
        or cursor:try_skip_next_and_surrounding_white_space(">>=")
        or cursor:try_skip_next_and_surrounding_white_space("**=")
    {
        return {start_position, start_position + 3};
    }

    if cursor:try_skip_next_and_surrounding_white_space("+=")
        or cursor:try_skip_next_and_surrounding_white_space("-=")
        or cursor:try_skip_next_and_surrounding_white_space("*=")
        or cursor:try_skip_next_and_surrounding_white_space("/=")
        or cursor:try_skip_next_and_surrounding_white_space("%=")
        or cursor:try_skip_next_and_surrounding_white_space("|=")
        or cursor:try_skip_next_and_surrounding_white_space("^=")
        or cursor:try_skip_next_and_surrounding_white_space("@=")
        or cursor:try_skip_next_and_surrounding_white_space("&=")
    {
        return {start_position, start_position + 2};
    }

    syntax_error(&mut cursor, sv("illegal assignment operator"));
    return {0, 0};
}

function get_next_ascii_only_identifier : (cursor : Cursor mut&) -> ProgramSlice = {
    if !ascii_is_text_token_start_char(cursor:get_char()) {
        syntax_error(&mut cursor, sv("identifier must start with an alphabetic character or an underscore"));
    }

    mut result : ProgramSlice  = cursor:slice(0);
    while !cursor:is_end_of_token() {
        cursor.position += 1;
        result.end_offset += 1;
    }
    // Assert length != 0
    return result;
}

function get_next_at_identifier : (cursor : Cursor mut&) -> ProgramSlice = {
    mut identifier : ProgramSlice = cursor:slice(1);
    runtime_assert(cursor:try_skip_next("@"));
    identifier.end_offset = cursor:get_next_ascii_only_identifier().end_offset;
    return identifier;
}

function parse_comma_separated_named_members : (cursor : Cursor mut&, matched_pair : StringView) -> Vector<TypedMember> = {
    let start_offset : isize = cursor.position;
    mut result : Vector<TypedMember> = make<Vector<TypedMember>>(&mut cursor.allocator);

    mut is_missing_comma : bool = false;
    while !cursor:try_skip_next_and_surrounding_white_space(matched_pair) {
        if cursor:get_char() == "\0"[0] {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated list, expected '%'"), matched_pair);
        }
        if is_missing_comma {
            syntax_error(&mut cursor, sv("expected ',' or '%'"), matched_pair);
        }

        // Argument name
        let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
        cursor:expect_next_skipping_ws(":");

        // Argument type
        let type : Type = cursor:parse_type();
        // TODO: we should probably be able to deduce `TypedMember` here
        result:push_back<TypedMember>({name, type});

        // Argument separators
        is_missing_comma = !cursor:try_skip_next_and_surrounding_white_space(",");
    }
    return result;
}

function parse_generic_definition_list : (cursor : Cursor mut&) -> Vector<GenericDefinition> = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("["));

    mut generics : Vector<GenericDefinition> = make<Vector<GenericDefinition>>(&mut cursor.allocator);

    mut is_missing_comma : bool = false;
    while !cursor:try_skip_next_and_surrounding_white_space("]") {
        if cursor:get_char() == "\0"[0] {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated list, expected ']'"));
        }
        if is_missing_comma {
            syntax_error(&mut cursor, sv("expected ',' or ']'"));
        }

        // TODO: this is a little ugly, should be refactored
        mut type : int;
        mut name : ProgramSlice;
        if cursor:get_char() == "@"[0] {
            type = /* generic numeric */ 1;
            name = cursor:get_next_at_identifier();
        } else {
            type = /* generic type */ 0;
            name = cursor:get_next_ascii_only_identifier();
        }

        if cursor:try_skip_next_and_surrounding_white_space("...") {
            // We've encountered the parameter pack, this must be the last generic
            // TODO: infer this type
            generics:push_back<GenericDefinition>({type, name, true, meta(name.start_offset, cursor.position)});
            cursor:expect_next_skipping_ws("]");
            return generics;
        }

        generics:push_back<GenericDefinition>({type, name, false, meta(name.start_offset, cursor.position)});
        is_missing_comma = !cursor:try_skip_next_and_surrounding_white_space(",");
    }
    return generics;
}

function parse_named_type : (cursor : Cursor mut&) -> NamedType = {
    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    let specialization_list : Vector<Specialization> = cursor:parse_optional_specialization();
    return {name, specialization_list, meta(name.start_offset, cursor.position)};
}

function is_at_start_of_compile_time_constant : (cursor : Cursor mut&) -> bool = {
    return cursor:is_next("@") or ascii_is_digit(cursor:get_char()) or cursor:is_next("-");
}

function parse_compile_time_constant : (cursor : Cursor mut&) -> CompileTimeConstant = {
    runtime_assert(cursor:is_at_start_of_compile_time_constant());

    if cursor:is_next("@") {
        let identifier : ProgramSlice = cursor:get_next_at_identifier();
        return {true, identifier};
    } else {
        let number : NumericToken = cursor:get_next_numeric_token();
        if number.type == 1 {
            syntax_error(&mut cursor, sv("floating point literals cannot be used as a compile time constant"));
        }
        return {false, number.data};
    }
}

function parse_compile_time_constant_list : (cursor : Cursor mut&) -> Vector<CompileTimeConstant> = {
    // TODO: return some meta here
    // TODO: lets avoid this duplicate logic
    // Maybe we need function pointers
    let start_offset : isize = cursor.position;
    mut constants : Vector<CompileTimeConstant> = make<Vector<CompileTimeConstant>>(&mut cursor.allocator);

    // TODO: do while to avoid this duplication (which forces the list to have at least one element)
    if !cursor:is_at_start_of_compile_time_constant() {
        syntax_error(&mut cursor, sv("expected a compile time constant"));
    }

    mut is_missing_comma : bool = false;
    while !cursor:try_skip_next_and_surrounding_white_space("]") {
        if cursor:get_char() == "\0"[0] {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated list, expected ']'"));
        }
        if is_missing_comma {
            syntax_error(&mut cursor, sv("expected ',' or ']'"));
        }

        let constant : CompileTimeConstant = cursor:parse_compile_time_constant();
        constants:push_back(constant);

        if !cursor:try_skip_next_and_surrounding_white_space(",") {
            is_missing_comma = true;
        }
    }
    return constants;
}

function parse_optional_specialization : (cursor : Cursor mut&) -> Vector<Specialization> = {
    if cursor:is_next("<") {
        return cursor:parse_specialization_list();
    } else {
        return make<Vector<Specialization>>(&mut cursor.allocator);
    }
}

function parse_specialization_list : (cursor : Cursor mut&) -> Vector<Specialization> = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("<"));

    mut is_missing_comma : bool = false;

    mut specializations : Vector<Specialization> = make<Vector<Specialization>>(&mut cursor.allocator);
    while !cursor:try_skip_next_and_surrounding_white_space(">") {
        if cursor:get_char() == "\0"[0] {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated list, expected '>'"));
        }
        if is_missing_comma {
            syntax_error(&mut cursor, sv("expected ',' or ']'"));
        }

        if cursor:is_at_start_of_compile_time_constant() {
            let constant : CompileTimeConstant mut& = &mut cursor.allocator:allocate<CompileTimeConstant>();
            constant = cursor:parse_compile_time_constant();
            specializations:push_back<Specialization>({1, ref_to_addr(&constant)});
        } else {
            let type : Type mut& = &mut cursor.allocator:allocate<Type>();
            type = cursor:parse_type();
            specializations:push_back<Specialization>({0, ref_to_addr(&type)});
        }

        if !cursor:try_skip_next_and_surrounding_white_space(",") {
            is_missing_comma = true;
        }
    }
    return specializations;
}

function parse_struct_type : (cursor : Cursor mut&) -> StructType = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("{"));

    // TODO: trim whitespace from meta here
    let members : Vector<TypedMember> = cursor:parse_comma_separated_named_members(sv("}"));
    return {members, meta(start_offset, cursor.position)};
}


function parse_type_prefix : (cursor : Cursor mut&) -> Type = {
    let start_offset : isize = cursor.position;

    if ascii_is_text_token_start_char(cursor:get_char()) {
        let named_type : NamedType mut& = &mut cursor.allocator:allocate<NamedType>();
        named_type = cursor:parse_named_type();
        return {4, ref_to_addr(&named_type), meta(start_offset, named_type.meta.end_offset)};
    }

    // TODO: function types

    if cursor:is_next("{") {
        let struct_type : StructType mut& = &mut cursor.allocator:allocate<StructType>();
        struct_type = cursor:parse_struct_type();
        return {5, ref_to_addr(&struct_type), meta(start_offset, struct_type.meta.end_offset)};
    }

    syntax_error(&mut cursor, sv("unexpected symbol '%', this cannot be used at the start of a type"), decode_utf8(cursor:get_char()));

    // Unreachable
    // TODO: maybe allow `undefined` return
    return {0, 0, meta(0, 0)};
}

function parse_array_type : (cursor : Cursor mut&, prefix : Type) -> Type = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("["));

    let is_mutable : bool = cursor:try_skip_next_and_surrounding_white_space("mut");

    mut is_heap_array : bool;
    if is_mutable {
        // We've already seen a `mut` token, this MUST be a heap array
        is_heap_array = true;
        cursor:expect_next_skipping_ws("&");
    } else {
        is_heap_array = cursor:try_skip_next_and_surrounding_white_space("&");
    }

    mut sizes : Vector<CompileTimeConstant>;
    // A heap array may/ may not have further dimensions specified
    // TODO: I'm using `if ... {full} else {allocate empty}` pattern a lot... is there a nicer way
    if is_heap_array and !cursor:try_skip_next_and_surrounding_white_space(",") {
        sizes = make<Vector<CompileTimeConstant>>(&mut cursor.allocator);
        cursor:expect_next_skipping_ws("]");
    } else {
        // A stack array must have a known size
        sizes = cursor:parse_compile_time_constant_list();
    }

    let meta : Meta = {prefix.meta.start_offset, cursor.position};
    let array : ArrayType mut& = &mut cursor.allocator:allocate<ArrayType>();
    array = {prefix, sizes, is_heap_array, is_mutable, meta};

    let result_type : Type = {1, ref_to_addr(&array), meta};
    if cursor:is_next("[") {
        return cursor:parse_array_type(result_type);
    } else {
        return result_type;
    }
}

function parse_type : (cursor : Cursor mut&) -> Type = {
    mut prefix : Type = cursor:parse_type_prefix();

    if cursor:is_next("[") {
        prefix = cursor:parse_array_type(prefix);
    }

    if cursor:try_skip_next("...") {
        let pack_type : PackType mut& = &mut cursor.allocator:allocate<PackType>();
        let meta : Meta = meta(prefix.meta.start_offset, cursor:skip_ws_return_offset());
        pack_type = {prefix, meta};
        return {6, ref_to_addr(&pack_type), meta};
    }

    if cursor:try_skip_next("&") {
        let ref_type : RefType mut& = &mut cursor.allocator:allocate<RefType>();
        let meta : Meta = meta(prefix.meta.start_offset, cursor:skip_ws_return_offset());
        ref_type = {prefix, false, meta};
        return {2, ref_to_addr(&ref_type), meta};
    }

    if cursor:try_skip_next_and_surrounding_white_space("mut") {
        cursor:expect_next("&");
        let ref_type : RefType mut& = &mut cursor.allocator:allocate<RefType>();
        let meta : Meta = meta(prefix.meta.start_offset, cursor:skip_ws_return_offset());
        ref_type = {prefix, true, meta};
        return {2, ref_to_addr(&ref_type), meta};
    }

    // Otherwise there is no suffix modifying the type
    return prefix;
}

function parse_typedef : (cursor : Cursor mut&) -> Typedef = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("typedef"));

    let generics : Vector<GenericDefinition> = cursor:parse_optional_generic_definition_list();

    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();

    cursor:expect_next_skipping_ws(":");

    let type : Type = cursor:parse_type();

    return {name, generics, specialization, type, meta(name.start_offset, type.meta.end_offset)};
}

function parse_list_of_expressions : (cursor : Cursor mut&, end : StringView) -> Vector<Expression> = {
    mut result : Vector<Expression> = make<Vector<Expression>>(&mut cursor.allocator);

    let start_offset : isize = cursor.position;
    mut is_missing_comma : bool = false;
    while !cursor:try_skip_next_and_surrounding_white_space(end) {
        // TODO: I've done this logic like 5 times, function pointers would eliminate this
        if cursor:get_char() == "\0"[0] {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated list, expected '%'"), end);
        }
        if is_missing_comma {
            syntax_error(&mut cursor, sv("expected ',' or '%'"), end);
        }
        result:push_back(cursor:parse_expression());
        is_missing_comma = !cursor:try_skip_next_and_surrounding_white_space(",");
    }
    return result;
}

function parse_list_of_named_expressions : (cursor : Cursor mut&, end : StringView) -> Vector<NamedExpression> = {
    // TODO: return some meta here
    mut result : Vector<NamedExpression> = make<Vector<NamedExpression>>(&mut cursor.allocator);

    let start_offset : isize = cursor.position;
    mut is_missing_comma : bool = false;
    while !cursor:try_skip_next_and_surrounding_white_space(end) {
        if cursor:get_char() == "\0"[0] {
            syntax_error_at_position(&mut cursor, start_offset, sv("unterminated list, expected '%'"), end);
        }
        if is_missing_comma {
            syntax_error(&mut cursor, sv("expected ',' or '%'"), end);
        }

        cursor:expect_next(".");
        let name : ProgramSlice = cursor:get_next_ascii_only_identifier();

        cursor:expect_next_skipping_ws("=");

        let expression : Expression = cursor:parse_expression();
        result:push_back<NamedExpression>({name, expression});
        is_missing_comma = !cursor:try_skip_next_and_surrounding_white_space(",");
    }
    return result;
}

function parse_initializer_list : (cursor : Cursor mut&) -> Expression = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("{"));

    if cursor:is_next(".") {
        let named_expressions : Vector<NamedExpression> = cursor:parse_list_of_named_expressions(sv("}"));
        let memory : NamedInitializerList mut& = &mut cursor.allocator:allocate<NamedInitializerList>();
        let meta : Meta = meta(start_offset, cursor.position);

        memory = {named_expressions, meta};
        return {2, ref_to_addr(&memory), meta};
    } else {
        let expressions : Vector<Expression> = cursor:parse_list_of_expressions(sv("}"));
        let memory : UnnamedInitializerList mut& = &mut cursor.allocator:allocate<UnnamedInitializerList>();
        let meta : Meta = meta(start_offset, cursor.position);

        memory = {expressions, meta};
        return {3, ref_to_addr(&memory), meta};
    }
}

function parse_ufcs_call : (cursor : Cursor mut&, prefix : Expression) -> UFCS_Call = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space(":"));

    let fn_name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();

    cursor:expect_next_skipping_ws("(");
    let arguments : Vector<Expression> = cursor:parse_list_of_expressions(sv(")"));
    return {prefix, fn_name, specialization, arguments, meta(prefix.meta.start_offset, cursor.position)};
}

function parse_array_index_access : (cursor : Cursor mut&, prefix : Expression) -> ArrayIndexAccess = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("["));
    let indexes : Vector<Expression> = cursor:parse_list_of_expressions(sv("]"));
    return {prefix, indexes, meta(prefix.meta.start_offset, cursor.position)};
}

function parse_struct_index_access : (cursor : Cursor mut&, prefix : Expression) -> StructIndexAccess = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("."));

    let member_name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    return {prefix, member_name, meta(prefix.meta.start_offset, member_name.end_offset)};
}

function parse_callable_call : (cursor : Cursor mut&, prefix : Expression) -> FunctionCall = {
    // TODO: SUPPORT CALLABLES, correct signature!!!!!
    // atm. this is just a hacky way to call (non-specialized) functions
    runtime_assert(cursor:is_next("("));
    return cursor:parse_function_call(prefix);
}

function parse_function_call : (cursor : Cursor mut&, prefix : Expression) -> FunctionCall = {
    if prefix.type != /* variable */ 4 {
        syntax_error(&mut cursor, sv("callable types are not currently supported"));
    }

    mut fn_name : ProgramSlice;
    {
        // TODO: this is a temporary work around since we do not have proper callables
        let lhs_variable : VariableAccess& = &addr_to_ref<VariableAccess>(prefix.pointer);
        fn_name = lhs_variable.name;
        cursor.allocator:deallocate(&lhs_variable);
    }

    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();

    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("("));
    let arguments : Vector<Expression> = cursor:parse_list_of_expressions(sv(")"));
    return {fn_name, specialization, arguments, meta(prefix.meta.start_offset, cursor.position)};
}

function is_likely_the_start_of_a_specialized_function : (cursor : Cursor mut&) -> bool = {
    // This function does an approximate forward lookahead to determine whether
    // the cursor is currently at the start of a less than expression or a
    // specialized function call.

    // Since it does not do a full parse, it may incorrectly identify a legal
    // less than operator an illegal function call, the user can always correct
    // this with parens.

    if !cursor:is_next("<") {
        return false;
    }

    mut brackets_stack : Vector<u8> = make<Vector<u8>>(&mut cursor.allocator);
    mut lookahead : Cursor = cursor;
    while !lookahead:is_next("\0") {
        if is_one_of(lookahead:get_char(), "<"[0], "("[0], "{"[0], "["[0]) {
            brackets_stack:push_back(lookahead:get_char());
        }

        if is_one_of(lookahead:get_char(), ">"[0], ")"[0], "}"[0], "]"[0]) {
            if !ascii_is_matched_pair(brackets_stack:pop_back(), lookahead:get_char()) {
                brackets_stack:deallocate();
                return false;
            }

            // We've closed the final triangle bracket, does it still look like a function?
            if brackets_stack:length() == 0 {
                brackets_stack:deallocate();

                lookahead.position += 1;
                lookahead:try_skip_white_space();

                return lookahead:is_next("(");
            }
        }

        // Otherwise, advance the lookahead
        lookahead.position += 1;
        lookahead:try_skip_white_space();
    }

    // Worst case (not possible without a syntax error)
    // We've reached the end of the file without seeing a closing triangle bracket
    brackets_stack:deallocate();
    return false;
}

function parse_expression_postfix : (cursor : Cursor mut&, prefix : Expression) -> Expression = {
    if cursor:try_skip_next("...") {
        // TODO: is this right? Do we support expanding expressions?
        let memory : PackExpansion mut& = &mut cursor.allocator:allocate<PackExpansion>();
        memory = {prefix, meta(prefix.meta.start_offset, cursor:skip_ws_return_offset())};
        // TODO: Do we support postfixes after expansions?
        return cursor:parse_expression_postfix({10, ref_to_addr(&memory), memory.meta});
    }
    if cursor:is_next(":") {
        let memory : UFCS_Call mut& = &mut cursor.allocator:allocate<UFCS_Call>();
        memory = cursor:parse_ufcs_call(prefix);
        return cursor:parse_expression_postfix({7, ref_to_addr(&memory), memory.meta});
    }
    if cursor:is_next("[") {
        let memory : ArrayIndexAccess mut& = &mut cursor.allocator:allocate<ArrayIndexAccess>();
        memory = cursor:parse_array_index_access(prefix);
        return cursor:parse_expression_postfix({5, ref_to_addr(&memory), memory.meta});
    }
    if cursor:is_next(".") {
        let memory : StructIndexAccess mut& = &mut cursor.allocator:allocate<StructIndexAccess>();
        memory = cursor:parse_struct_index_access(prefix);
        return cursor:parse_expression_postfix({6, ref_to_addr(&memory), memory.meta});
    }
    if cursor:is_next("(") {
        // TODO: this should be a special callable type
        let memory : FunctionCall mut& = &mut cursor.allocator:allocate<FunctionCall>();
        memory = cursor:parse_callable_call(prefix);
        return cursor:parse_expression_postfix({0, ref_to_addr(&memory), memory.meta});
    }
    if cursor:is_likely_the_start_of_a_specialized_function() {
        let memory : FunctionCall mut& = &mut cursor.allocator:allocate<FunctionCall>();
        memory = cursor:parse_function_call(prefix);
        return cursor:parse_expression_postfix({0, ref_to_addr(&memory), memory.meta});
    }
    return prefix;
}

function parse_non_operator_expression : (cursor : Cursor mut&) -> Expression = {
    let prefix : Expression = cursor:parse_non_operator_expression_prefix();
    return cursor:parse_expression_postfix(prefix);
}

function parse_non_operator_expression_prefix : (cursor : Cursor mut&) -> Expression = {
    if cursor:try_skip_next_and_surrounding_white_space("(") {
        let expression : Expression = cursor:parse_expression();
        cursor:expect_next_skipping_ws(")");
        return expression;
    }

    // Constants
    if cursor:is_next("true") or cursor:is_next("false") {
        let constant : Constant mut& = &mut cursor.allocator:allocate<Constant>();
        constant = {3, cursor:get_next_bool_token()};
        return {1, ref_to_addr(&constant), get_meta(constant.value)};
    }
    if ascii_is_digit(cursor:get_char()) {
        let constant : Constant mut& = &mut cursor.allocator:allocate<Constant>();

        let token : NumericToken = cursor:get_next_numeric_token();
        constant = {token.type, token.data};
        return {1, ref_to_addr(&constant), get_meta(constant.value)};
    }
    if cursor:is_next("\"") {
        let constant : Constant mut& = &mut cursor.allocator:allocate<Constant>();
        constant = {0, cursor:get_next_string_literal_token()};
        return {1, ref_to_addr(&constant), get_meta(constant.value)};
    }
    if cursor:is_next("@") {
        let constant : Constant mut& = &mut cursor.allocator:allocate<Constant>();
        constant = {4, cursor:get_next_at_identifier()};
        return {1, ref_to_addr(&constant), get_meta(constant.value)};
    }

    // Everything else
    if cursor:is_next("{") {
        return cursor:parse_initializer_list();
    }

    if ascii_is_text_token_start_char(cursor:get_char()) {
        let variable_access : VariableAccess mut& = &mut cursor.allocator:allocate<VariableAccess>();

        let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
        variable_access = {name};
        return {4, ref_to_addr(&variable_access), get_meta(name)};
    }

    syntax_error(&mut cursor, sv("unexpected symbol '%', cannot start an expression"), decode_utf8(cursor:get_char()));
    return {0, 0, meta(0, 0)};
}

function parse_maybe_unary_or_borrow_expression : (cursor : Cursor mut&) -> Expression = {
    let start_offset : isize = cursor.position;

    if is_one_of(cursor:get_char(), "!"[0], "-"[0], "+"[0], "~"[0]) {
        let name : ProgramSlice = cursor:slice(1);
        cursor.position += 1;
        cursor:try_skip_white_space();

        let rhs : Expression = cursor:parse_maybe_unary_or_borrow_expression();

        let operator_call : UnaryOperator mut& = &mut cursor.allocator:allocate<UnaryOperator>();
        operator_call = {name, rhs, meta(start_offset, rhs.meta.end_offset)};

        return {9, ref_to_addr(&operator_call), operator_call.meta};
    }

    if cursor:try_skip_next("&") {
        let is_mut : bool = cursor:try_skip_next_and_surrounding_white_space("mut");
        let rhs : Expression = cursor:parse_maybe_unary_or_borrow_expression();

        let borrow : Borrow mut& = &mut cursor.allocator:allocate<Borrow>();
        borrow = {is_mut, rhs, meta(start_offset, borrow.meta.end_offset)};
        return {11, ref_to_addr(&borrow), borrow.meta};
    }

    return cursor:parse_non_operator_expression();
}

function [T] parse_next_operator_subexpression : (cursor: Cursor mut&, op_iter : T mut&) -> Expression = {
    if op_iter:has_next() {
        return cursor:parse_expression_with_order_of_operation(op_iter);
    } else {
        return cursor:parse_maybe_unary_or_borrow_expression();
    }
}

function [T] parse_expression_with_order_of_operation : (cursor : Cursor mut&, op_iter : T) -> Expression = {
    // Copy so that we can call the iterator functions
    mut op_iter_copy : T = op_iter;

    // TODO: make the pattern matching robust enough to put this in the standard library
    let current_op : StringView = op_iter_copy:__builtin_get_next();
    mut lhs : Expression = cursor:parse_next_operator_subexpression(&mut op_iter_copy);

    mut last_operator_start : isize = cursor.position;
    while cursor:try_skip_next_operator(current_op) {
        let op_name : ProgramSlice = {last_operator_start, cursor:skip_ws_return_offset()};

        let rhs : Expression = cursor:parse_next_operator_subexpression(&mut op_iter_copy);

        let meta : Meta = meta(lhs.meta.start_offset, cursor.position);

        if is_one_of(cursor:get_sv(op_name), sv("and"), sv("or")) {
            // This is currently very similar to the binary operator code.
            // However, binary operators will be parsed directly as function
            // calls in the future, so the code will diverge.
            let logical_operator : LogicalOperator mut& = &mut cursor.allocator:allocate<LogicalOperator>();
            logical_operator = {op_name, lhs, rhs, meta};
            lhs = {12, ref_to_addr(&logical_operator), meta};
        }
        else {
            let operator_call : Operator mut& = &mut cursor.allocator:allocate<Operator>();
            operator_call = {op_name, lhs, rhs, meta};
            lhs = {8, ref_to_addr(&operator_call), meta};
        }

        last_operator_start = cursor.position;
    }
    return lhs;
}

function parse_expression : (cursor : Cursor mut&) -> Expression = {
    // TODO: this size argument needs to somehow be automatic
    // TODO: all items on the same line should have the same priority
    let order_of_operation : StringView[21] = {
        sv("or"),
        sv("and"),
        sv("=="), sv("<"), sv(">"), sv(">="), sv("<="), sv("!="), sv("<=>"),
        sv("|"),
        sv("^"),
        sv("&"),
        sv("<<"), sv(">>"),
        sv("+"), sv("-"),
        sv("@"), sv("*"), sv("/"), sv("%"),
        sv("**"),
    };
    return cursor:parse_expression_with_order_of_operation(order_of_operation:get_iter());
}

function parse_either_assignment_or_expression : (cursor : Cursor mut&) -> Line = {
    let lhs : Expression = cursor:parse_expression();
    if cursor:try_skip_next_and_surrounding_white_space(";") {
        // We've reached the end of the line, this must just be an expression
        let expression_memory : Expression mut& = &mut cursor.allocator:allocate<Expression>();
        expression_memory = lhs;
        return {0, ref_to_addr(&expression_memory), lhs.meta};
    }

    // The line hasn't finished, this must be an assignment
    let assignment_operator : ProgramSlice = cursor:get_next_assignment();
    let rhs : Expression = cursor:parse_expression();

    let assignment_memory : Assignment mut& = &mut cursor.allocator:allocate<Assignment>();
    assignment_memory = {lhs, assignment_operator, rhs, meta(lhs.meta.start_offset, rhs.meta.end_offset)};

    cursor:expect_next(";");
    return {1, ref_to_addr(&assignment_memory), meta(lhs.meta.start_offset, cursor:skip_ws_return_offset())};
}

function parse_if : (cursor : Cursor mut&) -> If = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("if"));

    let condition : Expression = cursor:parse_expression();
    let if_scope : Scope = cursor:parse_scope();

    mut else_scope : Scope;
    if cursor:try_skip_next_and_surrounding_white_space("else") {
        else_scope = cursor:parse_scope();
    } else {
        // Default the scope to empty
        else_scope = {make<Vector<Line>>(&mut cursor.allocator), meta(if_scope.meta.end_offset, if_scope.meta.end_offset)};
    }
    return {condition, if_scope, else_scope, meta(start_offset, else_scope.meta.end_offset)};
}

function parse_while : (cursor : Cursor mut&) -> While = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("while"));

    let condition : Expression = cursor:parse_expression();
    let scope : Scope = cursor:parse_scope();
    return {condition, scope, meta(start_offset, scope.meta.end_offset)};
}

function parse_for : (cursor : Cursor mut&) -> For = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("for"));

    // TODO: accept type hints
    let variable_name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    cursor:expect_next_skipping_ws("in");

    let expression : Expression = cursor:parse_expression();
    let scope : Scope = cursor:parse_scope();
    return {variable_name, expression, scope, meta(start_offset, scope.meta.end_offset)};
}

function parse_let_declaration : (cursor : Cursor mut&) -> VariableDeclaration = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("let"));

    let variable_name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    cursor:expect_next_skipping_ws(":");

    let type : Type = cursor:parse_type();

    if !cursor:try_skip_next_and_surrounding_white_space("=") {
        syntax_error(&mut cursor, sv("a non-mutable value cannot be declared uninitialized"));
    }

    let expression : Expression = cursor:parse_expression();

    cursor:expect_next(";");
    return {false, variable_name, type, make<Optional<Expression>>(expression), meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_mut_declaration : (cursor : Cursor mut&) -> VariableDeclaration = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("mut"));

    let variable_name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    cursor:expect_next_skipping_ws(":");

    let type : Type = cursor:parse_type();

    mut expression_opt : Optional<Expression>;
    if cursor:try_skip_next_and_surrounding_white_space("=") {
        expression_opt = make<Optional<Expression>>(cursor:parse_expression());
    } else {
        expression_opt = make<Optional<Expression>>();
    }

    cursor:expect_next(";");
    return {true, variable_name, type, expression_opt, meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_return : (cursor : Cursor mut&) -> Return = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("return"));

    if cursor:try_skip_next_and_surrounding_white_space(";") {
        return {make<Optional<Expression>>(), meta(start_offset, cursor.position)};
    }

    let expression : Expression = cursor:parse_expression();
    cursor:expect_next(";");
    return {make<Optional<Expression>>(expression), meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_continue : (cursor : Cursor mut&) -> Continue = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("continue"));

    cursor:expect_next(";");
    return {meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_break : (cursor : Cursor mut&) -> Break = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("break"));

    cursor:expect_next(";");
    return {meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_line : (cursor : Cursor mut&) -> Line = {
    // TODO: this should be templated?
    cursor:try_skip_white_space();
    if cursor:is_next("if") {
        let pointer : If mut& = &mut cursor.allocator:allocate<If>();
        pointer = cursor:parse_if();
        return {3, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("while") {
        let pointer : While mut& = &mut cursor.allocator:allocate<While>();
        pointer = cursor:parse_while();
        return {4, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("for") {
        let pointer : For mut& = &mut cursor.allocator:allocate<For>();
        pointer = cursor:parse_for();
        return {5, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("return") {
        let pointer : Return mut& = &mut cursor.allocator:allocate<Return>();
        pointer = cursor:parse_return();
        return {2, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("continue") {
        let pointer : Continue mut& = &mut cursor.allocator:allocate<Continue>();
        pointer = cursor:parse_continue();
        return {8, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("break") {
        let pointer : Break mut& = &mut cursor.allocator:allocate<Break>();
        pointer = cursor:parse_break();
        return {9, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("{") {
        let pointer : Scope mut& = &mut cursor.allocator:allocate<Scope>();
        pointer = cursor:parse_scope();
        return {6, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("let") {
        let pointer : VariableDeclaration mut& = &mut cursor.allocator:allocate<VariableDeclaration>();
        pointer = cursor:parse_let_declaration();
        return {7, ref_to_addr(&pointer), pointer.meta};
    }
    if cursor:is_next("mut") {
        let pointer : VariableDeclaration mut& = &mut cursor.allocator:allocate<VariableDeclaration>();
        pointer = cursor:parse_mut_declaration();
        return {7, ref_to_addr(&pointer), pointer.meta};
    }

    return cursor:parse_either_assignment_or_expression();
}

function parse_scope : (cursor : Cursor mut&) -> Scope = {
    let start_offset : isize = cursor.position;
    cursor:expect_next_skipping_ws("{");

    mut lines : Vector<Line> = make<Vector<Line>>(&mut cursor.allocator);
    while !cursor:try_skip_next("}") {
        if cursor:get_char() == "\0"[0] {
            syntax_error(&mut cursor, sv("unterminated scope, expected '}'"));
        }
        lines:push_back(cursor:parse_line());
    }

    return {lines, meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_optional_generic_definition_list : (cursor : Cursor mut&) -> Vector<GenericDefinition> = {
    if cursor:is_next("[") {
        return cursor:parse_generic_definition_list();
    } else {
        return make<Vector<GenericDefinition>>(&mut cursor.allocator);
    }
}

function parse_function_definition_args : (cursor : Cursor mut&) -> Vector<TypedMember> = {
    cursor:expect_next_skipping_ws(":");
    cursor:expect_next_skipping_ws("(");
    return cursor:parse_comma_separated_named_members(sv(")"));
}

function parse_function_definition_return_type : (cursor : Cursor mut&) -> Type = {
    cursor:expect_next_skipping_ws("->");
    return cursor:parse_type();
}

function parse_function_body : (cursor : Cursor mut&) -> Scope = {
    cursor:expect_next_skipping_ws("=");
    return cursor:parse_scope();
}

function parse_graphene_function_definition : (cursor : Cursor mut&) -> FunctionDefinition = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("function"));

    let generics : Vector<GenericDefinition> = cursor:parse_optional_generic_definition_list();
    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();
    let args : Vector<TypedMember> = cursor:parse_function_definition_args();
    let return_type : Type = cursor:parse_function_definition_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {0, generics, name, specialization, args, return_type, scope, meta(start_offset, scope.meta.end_offset)};
}

function parse_implicit_function_definition : (cursor : Cursor mut&) -> FunctionDefinition = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@implicit"));

    let generics : Vector<GenericDefinition> = cursor:parse_optional_generic_definition_list();
    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    if !is_one_of(cursor:get_sv(name), sv("has_next"), sv("get_next"), sv("destruct")) {
        syntax_error_at_position(&mut cursor, name.start_offset, sv("only 'has_next', 'get_next', and 'destruct' are valid implicit function names"));
    }

    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();
    let args : Vector<TypedMember> = cursor:parse_function_definition_args();
    let return_type : Type = cursor:parse_function_definition_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {1, generics, name, specialization, args, return_type, scope, meta(start_offset, scope.meta.end_offset)};
}

function parse_operator_function_definition : (cursor : Cursor mut&) -> FunctionDefinition = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@operator"));

    let generics : Vector<GenericDefinition> = cursor:parse_optional_generic_definition_list();
    let name : ProgramSlice = cursor:get_next_operator();
    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();
    let args : Vector<TypedMember> = cursor:parse_function_definition_args();
    let return_type : Type = cursor:parse_function_definition_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {2, generics, name, specialization, args, return_type, scope, meta(start_offset, scope.meta.end_offset)};
}

function parse_assignment_function_definition : (cursor : Cursor mut&) -> FunctionDefinition = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@assignment"));

    let generics : Vector<GenericDefinition> = cursor:parse_optional_generic_definition_list();
    let name : ProgramSlice = cursor:get_next_assignment();
    let specialization : Vector<Specialization> = cursor:parse_optional_specialization();
    let args : Vector<TypedMember> = cursor:parse_function_definition_args();
    let return_type : Type = cursor:parse_function_definition_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {4, generics, name, specialization, args, return_type, scope, meta(start_offset, scope.meta.end_offset)};
}

function parse_foreign_function_definition : (cursor : Cursor mut&) -> FunctionDefinition = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("foreign"));

    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    let args : Vector<TypedMember> = cursor:parse_function_definition_args();
    let return_type : Type = cursor:parse_function_definition_return_type();

    let empty_specialization : Vector<Specialization> = make<Vector<Specialization>>(&mut cursor.allocator);
    let empty_generics : Vector<GenericDefinition> = make<Vector<GenericDefinition>>(&mut cursor.allocator);
    let empty_scope : Scope = { make<Vector<Line>>(&mut cursor.allocator) , meta(0, 0)};

    return {3, empty_generics, name, empty_specialization, args, return_type, empty_scope, meta(start_offset, return_type.meta.end_offset)};
}

function parse_require_once : (cursor : Cursor mut&) -> RequireOnce = {
    let start_offset : isize = cursor.position;
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@require_once"));

    let path : ProgramSlice = cursor:get_next_string_literal_token();
    return {path, meta(start_offset, cursor:skip_ws_return_offset())};
}

function parse_next_top_level_feature : (cursor : Cursor mut&) -> TopLevelFeature = {
    if cursor:is_next("typedef") {
        let memory : Typedef mut& = &mut cursor.allocator:allocate<Typedef>();
        memory = cursor:parse_typedef();
        return {0, ref_to_addr(&memory), memory.meta};
    }

    if cursor:is_next("@require_once") {
        let memory : RequireOnce mut& = &mut cursor.allocator:allocate<RequireOnce>();
        memory = cursor:parse_require_once();
        return {2, ref_to_addr(&memory), memory.meta};
    }

    {
        let memory : FunctionDefinition mut& = &mut cursor.allocator:allocate<FunctionDefinition>();

        if cursor:is_next("function") {
            memory = cursor:parse_graphene_function_definition();
            return {1, ref_to_addr(&memory), memory.meta};
        }
        if cursor:is_next("foreign") {
            memory = cursor:parse_foreign_function_definition();
            return {1, ref_to_addr(&memory), memory.meta};
        }
        if cursor:is_next("@implicit") {
            memory = cursor:parse_implicit_function_definition();
            return {1, ref_to_addr(&memory), memory.meta};
        }
        if cursor:is_next("@assignment") {
            memory = cursor:parse_assignment_function_definition();
            return {1, ref_to_addr(&memory), memory.meta};
        }
        if cursor:is_next("@operator") {
            memory = cursor:parse_operator_function_definition();
            return {1, ref_to_addr(&memory), memory.meta};
        }

        // TODO: maybe just copy-paste the allocation line to avoid unnecessary memory management
        cursor.allocator:deallocate(&memory);
    }

    syntax_error(&mut cursor, sv("expected one of the following top level program features:\\n - typedef\\n - @require_once\\n - function\\n - foreign\\n - @implicit\\n - @assignment\\n - @operator"));
    return {0, 0, meta(0, 0)};
}

function main : (argc: int, argv: u8[&][&]) -> int = {
    // 1) Read the entire file
    let file_to_parse : CString = {&argv[1]};
    let file : Optional<File> = open(file_to_parse, O_RDONLY());
    if !file:has_value() {
        printf(sv("Error: could not open file '%'"), make<StringView>(file_to_parse));
        return 1;
    }

    let maybe_map : Optional<MappedFile> = file:data():read();
    file:data():close();  // fd no longer needed
    if !maybe_map:has_value() {
        printf(sv("Error: could not map file '%'"), make<StringView>(file_to_parse));
        return 1;
    }
    let file_mapping : MappedFile& = &maybe_map:data();
    let program_text : StringView = { file_mapping.buffer };

    // 2) Start parsing!!
    mut allocator : Allocator = initialize_allocator();
    mut program : Vector<TopLevelFeature> = make<Vector<TopLevelFeature>>(&mut allocator);
    {
        mut cursor : Cursor = {program_text, 0, &mut allocator};
        cursor:try_skip_white_space();

        while !cursor:is_next("\0") {
            program:push_back(cursor:parse_next_top_level_feature());
        }
    }

    // 3) Output result as JSON
    {
        mut json_output : JSON_Stream = {
            .program = program_text,
            .line_number_lookup = generate_position_lookup_table(program_text, &mut allocator),
            .file = get_stdout_io(),
        };
        json_output:write_json_value(&program);
        json_output:close();
    }

    // 4) Cleanup
    file_mapping:unmap();

    return 0;
}

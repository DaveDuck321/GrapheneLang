@require_once "file_info.c3"
@require_once "std/algorithms.c3"
@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/iterators.c3"
@require_once "std/memory.c3"
@require_once "std/string.c3"
@require_once "std/syscalls.c3"
@require_once "std/vector.c3"

typedef TopLevelFeature : {
    type : int, // 0: typedef, 1: function, 2: require once
    pointer : iptr, // TODO: the indirection here is unneeded... a union would be better
}

typedef ProgramSlice : {
    start : isize,
    end : isize,
}

typedef CompileTimeConstant : {
    // TODO: variant
    is_identifier : bool,
    value : ProgramSlice,
}

typedef FunctionType : { /* TODO */ }
typedef ArrayType : { base_type : Type, size : Vector<CompileTimeConstant>, is_heap_array : bool }
typedef RefType : { type : Type }
typedef NamedType : { name : ProgramSlice, specialization : Vector<Specialization> }
typedef StructType : { members : Vector<TypedMember> }

typedef GenericDefinition : {
    // TODO: variant
    type : int, // 0: type, 1: numeric generic
    value : ProgramSlice,
}

typedef Specialization : {
    // TODO: variant
    type : int, // 0: type, 1: compile time constant
    pointer : iptr,
}

typedef Type : {
    // TODO: variant
    type: int, // 0: function, 1: array, 2: ref, 4: named type, 5: struct type
    pointer : iptr,
}

typedef TypedMember : {
    name : ProgramSlice,
    type : Type,
}

typedef FunctionDefinition : {
    // TODO: enum
    type : int, // 0: graphene, 1: implicit, 2: operator, 3: foreign, 4: assignment
    name : ProgramSlice,
    generic_definitions : Vector<GenericDefinition>,
    args : Vector<TypedMember>,
    return_type : Type,
    scope : Scope,
}

typedef RequireOnce : {
    file_path : ProgramSlice
}

typedef Typedef : {
    name : ProgramSlice,
    generic_definitions : Vector<GenericDefinition>,
    specialization : Vector<Specialization>,
    type : Type,
}

typedef Scope : { lines : Vector<Line> }
typedef Line : {
    // TODO: enum
    type : int, // 0: expression, 1: assignment, 2: return, 3: if, 4: while, 5: for, 6: scope
    pointer : iptr
}

typedef If : {
    condition : Expression,
    if_scope : Scope,
    else_scope : Scope,
}

typedef While : {
    condition : Expression,
    scope : Scope,
}

typedef For : {
    variable : ProgramSlice,
    iterator_expression : Expression,
    scope : Scope,
}

typedef Return : { expression : Expression }

typedef Assignment : {
    lhs : Expression,
    operator : ProgramSlice,
    rhs : Expression,
}
typedef Declaration : {
    is_const : bool,
    variable : ProgramSlice,
    type : Type,
    expression : Optional<Ptr<Expression>>,
}

typedef Expression : {
    // TODO: this is a poor mans variant
    // TODO: use an enum here

    // 0: function call, 1: constant, 2: named init list, 3: unnamed init list, 4: variable, 5: array index access, 6: struct index access, 7: ufcs
    type : int,
    pointer : iptr,
}

typedef FunctionCall : {
    name : ProgramSlice,
    arguments : Vector<Expression>,
}

typedef Constant : {
    // TODO: enum
    type : int, // 0: String, 1: float, 2: int, 3: bool, 4: generic identifier, 5: hex
    value : ProgramSlice,
}

typedef NamedInitializerList : {
    names : Vector<ProgramSlice>,
    values : Vector<Expression>,
}

typedef UnnamedInitializerList : {
    args : Vector<Expression>
}

typedef VariableName : {
    value : ProgramSlice
}

typedef ArrayIndexAccess : {
    expression : Expression,
    indicies: Vector<Expression>,
}

typedef StructIndexAccess : {
    expression : Expression,
    member : ProgramSlice,
}

typedef UFCS_Call : {
    expression : Expression,
    function_name : ProgramSlice,
    args : Vector<Expression>,
}

typedef Cursor : {
    program : StringView,
    position : isize,
    allocator: Allocator&, // TODO: this should be somewhere else?
}

function to_string_view : (slice : ProgramSlice&, program : StringView) -> StringView = {
    // TODO: there NEEDS to be a better way to do this
    let data : u8[&] = &addr_to_heap_array<u8>(ref_to_addr(&program.data[slice.start]));
    return {&data, slice.end - slice.start};
}

function ascii_is_text_token_start_char : (char : u8) -> bool = {
    return ascii_is_alpha(char) | (char == "_"[0]);
}

function ascii_is_text_token_char : (char : u8) -> bool = {
    return ascii_is_alphanumeric(char) | (char == "_"[0]);
}

function get_char : (cursor : Cursor&) -> u8 = {
    return cursor:get_char_at_offset(0);
}

function get_char_at_offset : (cursor : Cursor&, offset : isize) -> u8 = {
    if cursor.position + offset >= cursor.program.length {
        return "\0"[0];
    }
    // TODO: allow direct indexing into a stringview
    return cursor.program.data[cursor.position + offset];
}

function [@Len] is_next : (cursor : Cursor&, string : u8[@Len]&) -> bool = {
    return cursor:is_next(make_span(&string));
}

function is_next : (cursor : Cursor&, string : StringView) -> bool = {
    for i in range(string.length) {
        if cursor:get_char_at_offset(i) != string.data[i] {
            return false;
        }
    }
    return true;
}

function [@Len] try_skip_next_and_surrounding_white_space : (cursor : Cursor&, string : u8[@Len]&) -> bool = {
    return cursor:try_skip_next_and_surrounding_white_space(make_span(&string));
}

function try_skip_next_and_surrounding_white_space : (cursor : Cursor&, string : StringView) -> bool = {
    cursor:try_skip_white_space();
    let result : bool = cursor:try_skip_next(string);
    cursor:try_skip_white_space();
    return result;
}

function [@Len] try_skip_next : (cursor : Cursor&, string : u8[@Len]&) -> bool = {
    return cursor:try_skip_next(make_span(&string));
}

function try_skip_next : (cursor : Cursor&, string : StringView) -> bool = {
    for i in range(string.length) {
        if cursor:get_char_at_offset(i) != string.data[i] {
            return false;
        }
    }
    cursor.position += string.length;
    return true;
}

function [@Len] syntax_error : (cursor : Cursor&, description: u8[@Len]&) -> void = {
    syntax_error(&cursor, make_span(&description));
}

function syntax_error : (cursor : Cursor&, description: StringView) -> void = {
    let lookup_table : PositionLookupTable = generate_position_lookup_table(cursor.program, &cursor.allocator);
    let location : FilePosition = lookup_table:lookup_position(cursor.position);

    printf(make_span("Syntax error at offset: %, line : %, column : %\n"), location.offset, location.line_number, location.column_number);
    puts(description);

    runtime_assert(false);
}

function try_skip_comment : (cursor : Cursor&) -> bool = {
    // Single line comment
    if cursor:is_next("//") {
        while !cursor:is_next("\n") & !cursor:is_next("\0") {
            cursor.position += 1;
        }
        cursor.position += 1;
        return true;
    }

    // Multi-line comment
    if cursor:is_next("/*") {
        while !cursor:is_next("*/") & !cursor:is_next("\0") {
            cursor.position += 1;
        }
        cursor.position += 2;
        return true;
    }

    // Nothing to skip
    return false;
}

function try_skip_ascii_white_space : (cursor : Cursor&) -> bool = {
    let has_skipped : bool = false;
    while ascii_is_white_space(cursor:get_char()) {
        cursor.position += 1;
        has_skipped = true;
    }
    return has_skipped;
}

function try_skip_white_space : (cursor : Cursor&) -> bool = {
    // TODO: `do while` or similar
    let has_skipped_this_loop : bool = true;
    let has_skipped_any : bool = false;

    while has_skipped_this_loop {
        has_skipped_this_loop = false;
        has_skipped_this_loop |= cursor:try_skip_ascii_white_space();
        has_skipped_this_loop |= cursor:try_skip_comment();
        has_skipped_any |= has_skipped_this_loop;
    }
    return has_skipped_any;
}

function is_end_of_token : (cursor : Cursor&) -> bool = {
    if cursor:try_skip_white_space() {
        // White space can always end a token
        return true;
    }

    if !ascii_is_text_token_char(cursor:get_char()) {
        return true;
    }

    return false;
}

function get_next_hex_token : (cursor : Cursor&) -> ProgramSlice = {
    runtime_assert(cursor:is_next("0x"));

    let token : ProgramSlice = {cursor.position, cursor.position+2};
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
            token.end += 1;
        } else {
            // Otherwise its a syntax error
            // TODO: format string
            syntax_error(&cursor, "TODO: Unexpected symbol {} in token {}");
        }
    }
    // TODO: check that the token's length != 2
    return token;
}

function get_next_decimal_token : (cursor : Cursor&) -> ProgramSlice = {
    let token : ProgramSlice = {cursor.position, cursor.position};

    while !cursor:is_end_of_token() {
        let next_char : u8 = ascii_to_lower(cursor:get_char());
        if ascii_is_digit(next_char) {
            cursor.position += 1;
            token.end += 1;
        } else {
            // Otherwise its a syntax error
            // TODO: format string
            syntax_error(&cursor, "TODO: Unexpected symbol {} in token {}");
        }
    }
    // TODO: check that the token's length != 0
    return token;
}

function get_next_numeric_token : (cursor : Cursor&) -> ProgramSlice = {
    let token : ProgramSlice = {cursor.position, cursor.position};

    // Is this number hex?
    // TODO: binary/ octal
    if cursor:is_next("0x") {
        return cursor:get_next_hex_token();
    }
    return cursor:get_next_decimal_token();
}

function get_next_string_literal_token : (cursor : Cursor&) -> ProgramSlice = {
    runtime_assert(cursor:get_char() == "\""[0]);
    let start_offset : isize = cursor.position;
    cursor.position += 1;

    let current_char : u8 = cursor:get_char();
    while current_char != "\""[0] {
        if current_char == "\0"[0] {
            syntax_error(&cursor, "TODO: Unterminated string");
        }

        if current_char == "\\"[0] {
            // Skip the character after an escape sequence
            // TODO: allow numeric escape sequences eg \789
            cursor.position += 1;
        }
        cursor.position += 1;
        current_char = cursor:get_char();
    }
    // The string has been terminated
    cursor.position += 1;
    return {start_offset, cursor.position};
}

function get_next_operator : (cursor : Cursor&) -> ProgramSlice = {
    let start_position : isize = cursor.position;

    if cursor:try_skip_next_and_surrounding_white_space("==")
        | cursor:try_skip_next_and_surrounding_white_space("<=")
        | cursor:try_skip_next_and_surrounding_white_space(">=")
        | cursor:try_skip_next_and_surrounding_white_space("!=")
        | cursor:try_skip_next_and_surrounding_white_space("<<")
        | cursor:try_skip_next_and_surrounding_white_space(">>")
        | cursor:try_skip_next_and_surrounding_white_space("**")
    {
        return {start_position, start_position + 2};
    }

    if cursor:try_skip_next_and_surrounding_white_space("<")
        | cursor:try_skip_next_and_surrounding_white_space(">")
        | cursor:try_skip_next_and_surrounding_white_space("!")
        | cursor:try_skip_next_and_surrounding_white_space("~")
        | cursor:try_skip_next_and_surrounding_white_space("+")
        | cursor:try_skip_next_and_surrounding_white_space("-")
        | cursor:try_skip_next_and_surrounding_white_space("*")
        | cursor:try_skip_next_and_surrounding_white_space("/")
        | cursor:try_skip_next_and_surrounding_white_space("%")
        | cursor:try_skip_next_and_surrounding_white_space("|")
        | cursor:try_skip_next_and_surrounding_white_space("^")
        | cursor:try_skip_next_and_surrounding_white_space("@")
        | cursor:try_skip_next_and_surrounding_white_space("&")
    {
        return {start_position, start_position + 1};
    }

    syntax_error(&cursor, "TODO");
    return {0, 0};
}

function get_next_assignment : (cursor : Cursor&) -> ProgramSlice = {
    let start_position : isize = cursor.position;

    if cursor:try_skip_next_and_surrounding_white_space("=") {
        return {start_position, start_position + 1};
    }

    if cursor:try_skip_next_and_surrounding_white_space("<<=")
        | cursor:try_skip_next_and_surrounding_white_space(">>=")
        | cursor:try_skip_next_and_surrounding_white_space("**=")
    {
        return {start_position, start_position + 3};
    }

    if cursor:try_skip_next_and_surrounding_white_space("+=")
        | cursor:try_skip_next_and_surrounding_white_space("-=")
        | cursor:try_skip_next_and_surrounding_white_space("*=")
        | cursor:try_skip_next_and_surrounding_white_space("/=")
        | cursor:try_skip_next_and_surrounding_white_space("%=")
        | cursor:try_skip_next_and_surrounding_white_space("|=")
        | cursor:try_skip_next_and_surrounding_white_space("^=")
        | cursor:try_skip_next_and_surrounding_white_space("@=")
        | cursor:try_skip_next_and_surrounding_white_space("&=")
    {
        return {start_position, start_position + 2};
    }

    syntax_error(&cursor, "TODO");
    return {0, 0};
}


function get_next_ascii_only_identifier : (cursor : Cursor&) -> ProgramSlice = {
    if !ascii_is_text_token_start_char(cursor:get_char()) {
        syntax_error(&cursor, "TODO: Expected an alphabetic character");
    }

    let result : ProgramSlice  = {cursor.position, cursor.position};
    while !cursor:is_end_of_token() {
        if !ascii_is_text_token_char(cursor:get_char()) {
            syntax_error(&cursor, "TODO: Expected an alphanumeric character");
        }

        cursor.position += 1;
        result.end += 1;
    }
    // Assert length != 0
    return result;
}

function get_next_at_identifier : (cursor : Cursor&) -> ProgramSlice = {
    runtime_assert(cursor:try_skip_next("@"));
    let identifier : ProgramSlice = cursor:get_next_ascii_only_identifier();
    identifier.start -= 1;
    return identifier;
}

function parse_comma_separated_named_members : (cursor : Cursor&, matched_pair : StringView) -> Vector<TypedMember> = {
    let result : Vector<TypedMember> = make_vector<TypedMember>(&cursor.allocator);

    let is_missing_comma : bool = false;

    while !cursor:try_skip_next_and_surrounding_white_space(matched_pair) {
        if cursor:get_char() == "\0"[0] {
            syntax_error(&cursor, "Unterminated comma separated list");
        }
        if is_missing_comma {
            syntax_error(&cursor, "TODO: missing comma");
        }

        // Argument name
        let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
        if !cursor:try_skip_next_and_surrounding_white_space(":") {
            syntax_error(&cursor, "TODO: missing ':'");
        }

        // Argument type
        let type : Type = cursor:parse_type();
        // TODO: we should probably be able to deduce `TypedMember` here
        result:push_back<TypedMember>({name, type});

        // Argument separators
        if !cursor:try_skip_next_and_surrounding_white_space(",") {
            is_missing_comma = true;
        }
    }
    return result;
}

function parse_generic_definition_list : (cursor : Cursor&) -> Vector<GenericDefinition> = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("["));

    let is_missing_comma : bool = false;

    let generics : Vector<GenericDefinition> = make_vector<GenericDefinition>(&cursor.allocator);
    while cursor:try_skip_next_and_surrounding_white_space("]") {
        if cursor:get_char() == "\0"[0] {
            syntax_error(&cursor, "Unterminated generic definition list");
        }
        if is_missing_comma {
            syntax_error(&cursor, "TODO: missing comma");
        }

        if cursor:get_char() == "@"[0] {
            // TODO: deduce the type here
            generics:push_back<GenericDefinition>({
                1 /* generic numeric */,
                cursor:get_next_at_identifier(),
            });
        } else {
            generics:push_back<GenericDefinition>({
                0 /* generic type */,
                cursor:get_next_ascii_only_identifier(),
            });
        }

        if !cursor:try_skip_next_and_surrounding_white_space(",") {
            is_missing_comma = true;
        }
    }
    return generics;
}

function parse_named_type : (cursor : Cursor&) -> NamedType = {
    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    let specialization_list : Vector<Specialization>;
    if cursor:is_next("<") {
        specialization_list = cursor:parse_specialization_list();
    } else {
        specialization_list = make_vector<Specialization>(&cursor.allocator);
    }

    return {name, specialization_list};
}

function is_at_start_of_compile_time_constant : (cursor : Cursor&) -> bool = {
    return cursor:is_next("@") | ascii_is_digit(cursor:get_char());
}

function parse_compile_time_constant : (cursor : Cursor&) -> CompileTimeConstant = {
    runtime_assert(cursor:is_at_start_of_compile_time_constant());

    if cursor:is_next("@") {
        let identifier : ProgramSlice = cursor:get_next_at_identifier();
        return {true, identifier};
    } else {
        let number : ProgramSlice = cursor:get_next_numeric_token();
        return {false, number};
    }
}

function parse_compile_time_constant_list : (cursor : Cursor&) -> Vector<CompileTimeConstant> = {
    // TODO: lets avoid this duplicate logic
    // Maybe we need function pointers
    let is_missing_comma : bool = false;

    let constants : Vector<CompileTimeConstant> = make_vector<CompileTimeConstant>(&cursor.allocator);
    while cursor:try_skip_next_and_surrounding_white_space("]") {
        if cursor:get_char() == "\0"[0] {
            syntax_error(&cursor, "Unterminated constant list");
        }
        if is_missing_comma {
            syntax_error(&cursor, "TODO: missing comma");
        }

        let constant : CompileTimeConstant = cursor:parse_compile_time_constant();
        constants:push_back(constant);

        if !cursor:try_skip_next_and_surrounding_white_space(",") {
            is_missing_comma = true;
        }
    }
    return constants;
}

function parse_specialization_list : (cursor : Cursor&) -> Vector<Specialization> = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("<"));

    let is_missing_comma : bool = false;

    let specializations : Vector<Specialization> = make_vector<Specialization>(&cursor.allocator);
    while !cursor:try_skip_next_and_surrounding_white_space(">") {
        if cursor:get_char() == "\0"[0] {
            syntax_error(&cursor, "Unterminated specialization list");
        }
        if is_missing_comma {
            syntax_error(&cursor, "TODO: missing comma");
        }

        if cursor:is_at_start_of_compile_time_constant() {
            let constant : CompileTimeConstant& = &cursor.allocator:allocate<CompileTimeConstant>();
            constant = cursor:parse_compile_time_constant();
            specializations:push_back<Specialization>({1, ref_to_addr(&constant)});
        } else {
            let type : Type& = &cursor.allocator:allocate<Type>();
            type = cursor:parse_type();
            specializations:push_back<Specialization>({1, ref_to_addr(&type)});
        }

        if !cursor:try_skip_next_and_surrounding_white_space(",") {
            is_missing_comma = true;
        }
    }
    return specializations;
}

function parse_struct_type : (cursor : Cursor&) -> StructType = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("{"));

    let members : Vector<TypedMember> = cursor:parse_comma_separated_named_members(make_span("}"));
    return {members};
}


function parse_type_prefix : (cursor : Cursor&) -> Type = {
    if ascii_is_text_token_start_char(cursor:get_char()) {
        let named_type : NamedType& = &cursor.allocator:allocate<NamedType>();
        named_type = cursor:parse_named_type();
        return {4, ref_to_addr(&named_type)};
    }

    // TODO: function types

    if cursor:is_next("{") {
        let struct_type : StructType& = &cursor.allocator:allocate<StructType>();
        struct_type = cursor:parse_struct_type();
        return {5, ref_to_addr(&struct_type)};
    }


    syntax_error(&cursor, "TODO: Unexpected symbol {}");

    // Unreachable
    // TODO: maybe allow `undefined` return
    return {0, 0};
}

function parse_array_type : (cursor : Cursor&, prefix : Type) -> Type = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("["));

    let is_heap_array : bool = cursor:try_skip_next_and_surrounding_white_space("&");
    let sizes : Vector<CompileTimeConstant> = cursor:parse_compile_time_constant_list();

    let array : ArrayType& = &cursor.allocator:allocate<ArrayType>();
    array = {prefix, sizes, is_heap_array};

    let result_type : Type = {1, ref_to_addr(&array)};
    if cursor:is_next("[") {
        return cursor:parse_array_type(result_type);
    } else {
        return result_type;
    }
}

function parse_type : (cursor : Cursor&) -> Type = {
    let prefix : Type = cursor:parse_type_prefix();

    if cursor:is_next("[") {
        prefix = cursor:parse_array_type(prefix);
    }

    if cursor:try_skip_next_and_surrounding_white_space("&") {
        let ref_type : RefType& = &cursor.allocator:allocate<RefType>();
        ref_type = {prefix};
        return {2, ref_to_addr(&ref_type)};
    }

    // Otherwise there is no suffix modifying the type
    return prefix;
}

function parse_typedef : (cursor : Cursor&) -> Typedef = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("typedef"));

    let generics : Vector<GenericDefinition>;
    if cursor:is_next("[") {
        generics = cursor:parse_generic_definition_list();
    } else {
        generics = make_vector<GenericDefinition>(&cursor.allocator);
    }

    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    if !cursor:try_skip_next_and_surrounding_white_space(":") {
        syntax_error(&cursor, "TODO: expected a ':'");
    }

    // TODO: I'm using `if ... {full} else {allocate empty}` pattern a lot... is there a nicer way
    let specialization : Vector<Specialization>;
    if cursor:is_next("<") {
        specialization = cursor:parse_specialization_list();
    } else {
        specialization = make_vector<Specialization>(&cursor.allocator);
    }

    return {name, generics, specialization, cursor:parse_type()};
}

function parse_expression : (cursor : Cursor&) -> Expression = {
    runtime_assert(false);
    return {0, 0};
}

function parse_if : (cursor : Cursor&) -> If = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("if"));
    let condition : Expression = cursor:parse_expression();
    let if_scope : Scope = cursor:parse_scope();

    let else_scope : Scope;
    if cursor:try_skip_next_and_surrounding_white_space("else") {
        else_scope = cursor:parse_scope();
    } else {
        // Default the scope to empty
        else_scope = {make_vector<Line>(&cursor.allocator)};
    }
    return {condition, if_scope, else_scope};
}

function parse_while : (cursor : Cursor&) -> While = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("while"));
    let condition : Expression = cursor:parse_expression();
    let scope : Scope = cursor:parse_scope();
    return {condition, scope};
}

function parse_for : (cursor : Cursor&) -> For = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("for"));
    // TODO: accept type hints
    let variable_name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    if !cursor:try_skip_next_and_surrounding_white_space("in") {
        syntax_error(&cursor, "TODO: expected `in`");
    }

    let expression : Expression = cursor:parse_expression();
    let scope : Scope = cursor:parse_scope();
    return {variable_name, expression, scope};
}

function parse_return : (cursor : Cursor&) -> Return = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("return"));
    let expression : Expression = cursor:parse_expression();
    return {expression};
}

function parse_line : (cursor : Cursor&) -> Line = {
    // TODO: this should be templated?
    cursor:try_skip_white_space();
    if cursor:is_next("if") {
        let pointer : If& = &cursor.allocator:allocate<If>();
        pointer = cursor:parse_if();
        return {3, ref_to_addr(&pointer)};
    }
    if cursor:is_next("while") {
        let pointer : While& = &cursor.allocator:allocate<While>();
        pointer = cursor:parse_while();
        return {4, ref_to_addr(&pointer)};
    }
    if cursor:is_next("for") {
        let pointer : For& = &cursor.allocator:allocate<For>();
        pointer = cursor:parse_for();
        return {5, ref_to_addr(&pointer)};
    }
    if cursor:is_next("return") {
        let pointer : Return& = &cursor.allocator:allocate<Return>();
        pointer = cursor:parse_return();
        return {2, ref_to_addr(&pointer)};
    }
    if cursor:is_next("{") {
        let pointer : Scope& = &cursor.allocator:allocate<Scope>();
        pointer = cursor:parse_scope();
        return {6, ref_to_addr(&pointer)};
    }

    // The line is not a statement, it must be an expression or a assignment
    // TODO:
    runtime_assert(false);
    return {0, 0};
}

function parse_scope : (cursor : Cursor&) -> Scope = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("{"));

    let result : Scope = {make_vector<Line>(&cursor.allocator)};
    while !cursor:try_skip_next_and_surrounding_white_space("}") {
        if cursor:get_char() == "\0"[0] {
            syntax_error(&cursor, "TODO: unterminated '}'");
        }
        result.lines:push_back(cursor:parse_line());
    }
    return result;
}

function parse_function_generics : (cursor : Cursor&) -> Vector<GenericDefinition> = {
    let generics : Vector<GenericDefinition>; // Uninitialized

    if cursor:is_next("[") {
        generics = cursor:parse_generic_definition_list();
    } else {
        // Empty the vector if the function has no generic arguments
        generics = make_vector<GenericDefinition>(&cursor.allocator);
    }
    return generics;
}

function parse_function_args : (cursor : Cursor&) -> Vector<TypedMember> = {
    if !cursor:try_skip_next_and_surrounding_white_space(":") {
        syntax_error(&cursor, "TODO: expected a ':'");
    }

    if !cursor:try_skip_next_and_surrounding_white_space("(") {
        syntax_error(&cursor, "TODO: expected a '('");
    }

    return cursor:parse_comma_separated_named_members(make_span(")"));
}

function parse_function_return_type : (cursor : Cursor&) -> Type = {
    if !cursor:try_skip_next_and_surrounding_white_space("->") {
        syntax_error(&cursor, "TODO: expected '->'");
    }
    return cursor:parse_type();
}

function parse_function_body : (cursor : Cursor&) -> Scope = {
    if !cursor:try_skip_next_and_surrounding_white_space("=") {
        syntax_error(&cursor, "TODO: expected '='");
    }
    return cursor:parse_scope();
}

function parse_graphene_function : (cursor : Cursor&) -> FunctionDefinition = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("function"));

    let generics : Vector<GenericDefinition> = cursor:parse_function_generics();
    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();
    let args : Vector<TypedMember> = cursor:parse_function_args();
    let return_type : Type = cursor:parse_function_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {0, name, generics, args, return_type, scope};
}

function parse_implicit_function : (cursor : Cursor&) -> FunctionDefinition = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@implicit"));

    let generics : Vector<GenericDefinition> = cursor:parse_function_generics();
    let name : ProgramSlice = cursor:get_next_ascii_only_identifier();

    if !is_one_of(name:to_string_view(cursor.program), make_span("has_next"), make_span("get_next"), make_span("destruct")) {
        syntax_error(&cursor, "TODO: Only 'has_next', 'get_next', and 'destruct' are valid implicit functions");
    }

    let args : Vector<TypedMember> = cursor:parse_function_args();
    let return_type : Type = cursor:parse_function_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {1, name, generics, args, return_type, scope};
}

function parse_operator_function : (cursor : Cursor&) -> FunctionDefinition = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@operator"));

    let generics : Vector<GenericDefinition> = cursor:parse_function_generics();
    let name : ProgramSlice = cursor:get_next_operator();
    let args : Vector<TypedMember> = cursor:parse_function_args();
    let return_type : Type = cursor:parse_function_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {2, name, generics, args, return_type, scope};
}

function parse_assignment_function : (cursor : Cursor&) -> FunctionDefinition = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@assignment"));

    let generics : Vector<GenericDefinition> = cursor:parse_function_generics();
    let name : ProgramSlice = cursor:get_next_assignment();
    let args : Vector<TypedMember> = cursor:parse_function_args();
    let return_type : Type = cursor:parse_function_return_type();
    let scope : Scope = cursor:parse_function_body();

    return {4, name, generics, args, return_type, scope};
}

function parse_foreign_function : (cursor : Cursor&) -> FunctionDefinition = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("foreign"));

    let name : ProgramSlice = cursor:get_next_assignment();
    let args : Vector<TypedMember> = cursor:parse_function_args();
    let return_type : Type = cursor:parse_function_return_type();

    let empty_generics : Vector<GenericDefinition> = make_vector<GenericDefinition>(&cursor.allocator);
    let empty_scope : Scope = { make_vector<Line>(&cursor.allocator) };

    return {3, name, empty_generics, args, return_type, empty_scope};
}

function parse_require_once : (cursor : Cursor&) -> RequireOnce = {
    runtime_assert(cursor:try_skip_next_and_surrounding_white_space("@require_once"));

    let path : ProgramSlice = cursor:get_next_string_literal_token();
    cursor:try_skip_white_space();
    return {path};
}

function parse_next_top_level_feature : (cursor : Cursor&) -> TopLevelFeature = {
    if cursor:is_next("typedef") {
        let memory : Typedef& = &cursor.allocator:allocate<Typedef>();
        memory = cursor:parse_typedef();
        return {0, ref_to_addr(&memory)};
    }

    if cursor:is_next("@require_once") {
        let memory : RequireOnce& = &cursor.allocator:allocate<RequireOnce>();
        memory = cursor:parse_require_once();
        return {2, ref_to_addr(&memory)};
    }


    {
        let memory : FunctionDefinition& = &cursor.allocator:allocate<FunctionDefinition>();

        // function
        if cursor:is_next("function") {
            memory = cursor:parse_graphene_function();
            return {1, ref_to_addr(&memory)};
        }

        // foreign
        if cursor:is_next("foreign") {
            memory = cursor:parse_foreign_function();
            return {1, ref_to_addr(&memory)};
        }

        // @implicit
        if cursor:is_next("@implicit") {
            memory = cursor:parse_implicit_function();
            return {1, ref_to_addr(&memory)};
        }

        // @assignment
        if cursor:is_next("@assignment") {
            memory = cursor:parse_assignment_function();
            return {1, ref_to_addr(&memory)};
        }

        // @operator
        if cursor:is_next("@operator") {
            memory = cursor:parse_graphene_function();
            return {1, ref_to_addr(&memory)};
        }

        // TODO: maybe just copy-paste the allocation line to avoid unnecessary memory management
        cursor.allocator:deallocate(&memory);
    }

    syntax_error(&cursor, "Top level features must be one of TODO");
    return {0, 0};
}

function main : (argc: int, argv: u8[&][&]) -> int = {
    let file_to_parse : CString = {&argv[1]};
    let file : File = open(file_to_parse, SYS_O_RDONLY());


    let allocator : Allocator = initialize_allocator();
    let file_contents : Vector<u8> = make_vector<u8>(&allocator);

    // 1) Read the entire file
    {
        let read_buffer : u8[1024];

        // TODO: `do while` to avoid temp value? Or `break` to make this prettier
        let read_data : StringView = make_span("uninitialized");
        while read_data.length != 0 {
            read_data = file:read(make_span(&read_buffer));
            file_contents:extend_back(read_data);
        }
    }

    let program_text : StringView = file_contents:data();

    // 2) Start parsing!!
    let cursor : Cursor = {program_text, 0, &allocator};

    let program : Vector<TopLevelFeature> = make_vector<TopLevelFeature>(&allocator);
    while !cursor:is_next("\0") {
        puts("here!");
        program:push_back(cursor:parse_next_top_level_feature());
    }

    file_contents:deallocate();
    return 0;
}

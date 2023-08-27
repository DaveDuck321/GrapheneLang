@require_once "file_info.c3"
@require_once "std/algorithms.c3"
@require_once "std/format.c3"
@require_once "std/io.c3"
@require_once "std/string.c3"
@require_once "std/vector.c3"

typedef Meta : {
    start_offset : isize,
    end_offset : isize,
}

typedef ProgramSlice : {
    start_offset : isize,
    end_offset : isize,
}

typedef TopLevelFeature : {
    type : int, // 0: typedef, 1: function, 2: require once
    pointer : iptr, // TODO: the indirection here is unneeded... a union would be better
    meta : Meta,
}

typedef CompileTimeConstant : {
    // TODO: variant
    is_identifier : bool,
    value : ProgramSlice,
}

typedef FunctionType : { /* TODO */ }
typedef ArrayType : { base_type : Type, size : Vector<CompileTimeConstant>, is_heap_array : bool, meta : Meta}
typedef RefType : { type : Type, meta : Meta }
typedef NamedType : { name : ProgramSlice, specialization : SpecializationList, meta : Meta}
typedef StructType : { members : Vector<TypedMember>, meta : Meta}
typedef PackType : { type : Type, meta : Meta}

typedef GenericDefinition : {
    // TODO: variant
    type : int, // 0: type, 1: numeric generic
    value : ProgramSlice,
    is_packed : bool,
    meta : Meta,
}

typedef Specialization : {
    // TODO: variant
    type : int, // 0: type, 1: compile time constant
    pointer : iptr,
    meta : Meta,
}

typedef Type : {
    // TODO: variant
    type: int, // 0: function, 1: array, 2: ref, 4: named type, 5: struct type, 6: parameter pack
    pointer : iptr,
    meta : Meta
}

typedef TypedMember : {
    name : ProgramSlice,
    type : Type,
}

typedef FunctionDefinition : {
    // TODO: enum
    type : int, // 0: graphene, 1: implicit, 2: operator, 3: foreign, 4: assignment
    name : ProgramSlice,
    generic_definitions : GenericDefinitionList,
    args : FunctionDefinitionArguments,
    return_type : Type,
    scope : Scope,
    meta : Meta,
}

typedef RequireOnce : {
    file_path : ProgramSlice,
    meta : Meta,
}

typedef Typedef : {
    name : ProgramSlice,
    generic_definitions : GenericDefinitionList,
    specialization : SpecializationList,
    type : Type,
    meta : Meta,
}

typedef Scope : { lines : Vector<Line>, meta : Meta }
typedef Line : {
    // TODO: enum
    type : int, // 0: expression, 1: assignment, 2: return, 3: if, 4: while, 5: for, 6: scope, 7: declaration
    pointer : iptr,
    meta : Meta,
}

typedef If : {
    condition : Expression,
    if_scope : Scope,
    else_scope : Scope,
    meta : Meta,
}

typedef While : {
    condition : Expression,
    scope : Scope,
    meta : Meta,
}

typedef For : {
    variable : ProgramSlice,
    iterator_expression : Expression,
    scope : Scope,
    meta : Meta,
}

typedef Return : { expression : Optional<Expression>, meta : Meta }

typedef Assignment : {
    lhs : Expression,
    operator : ProgramSlice,
    rhs : Expression,
    meta : Meta,
}

typedef Declaration : {
    is_const : bool,
    variable : ProgramSlice,
    type : Type,
    expression : Optional<Expression>,
    meta : Meta,
}

typedef Expression : {
    // TODO: this is a poor man's variant
    // TODO: use an enum here

    // 0: function call, 1: constant, 2: named init list, 3: unnamed init list, 4: variable, 5: array index access, 6: struct index access, 7: ufcs, 8: operator, 9: unary operator, 10: pack expansion, 11: borrow
    type : int,
    pointer : iptr,
    meta : Meta,
}

typedef NamedExpression : {
    name : ProgramSlice,
    expression : Expression,
}

typedef Operator : {
    // TODO: this can probably be unified with the function call type
    name : ProgramSlice,
    lhs : Expression,
    rhs : Expression,
    meta : Meta,
}

typedef UnaryOperator : {
    name : ProgramSlice,
    rhs : Expression,
    meta : Meta,
}

typedef Borrow : {
    is_const : bool,
    expression : Expression,
    meta : Meta,
}

typedef FunctionCall : {
    name : ProgramSlice,
    specialization : SpecializationList,
    arguments : Vector<Expression>,
    meta : Meta,
}

typedef UFCS_Call : {
    expression : Expression,
    function_name : ProgramSlice,
    specialization : SpecializationList,
    args : Vector<Expression>,
    meta : Meta,
}

typedef PackExpansion : {
    expression : Expression,
    meta : Meta,
}

typedef Constant : {
    // TODO: enum
    type : int, // 0: String, 1: float, 2: int, 3: bool, 4: generic identifier, 5: hex
    value : ProgramSlice,
}

typedef NamedInitializerList : {
    args : Vector<NamedExpression>,
    meta : Meta,
}

typedef UnnamedInitializerList : {
    args : Vector<Expression>,
    meta : Meta,
}

typedef VariableAccess : {
    name : ProgramSlice,
}

typedef ArrayIndexAccess : {
    expression : Expression,
    indicies: Vector<Expression>,
    meta : Meta,
}

typedef StructIndexAccess : {
    expression : Expression,
    member : ProgramSlice,
    meta : Meta,
}

// Output
typedef JSON_Stream : {
    program : StringView, // Hacky global to support program slices
    line_number_lookup : PositionLookupTable,
    file : BufferedFile,
}

// The below are COMPLETELY UNNECESSARY... used to emulate lark tokens
typedef Null : {}
typedef NamedToken : {
    name : StringView,
    value : StringView,
    meta : Meta,
}

typedef Concatinate : {
    start : StringView,
    end : StringView,
}

typedef FunctionName : {
    name : ProgramSlice,
    specialization : Optional<SpecializationList>,
}

typedef GenericDefinitionList : {
    generics : Vector<GenericDefinition>,
    meta : Meta,
}

typedef SpecializationList : {
    specialization : Vector<Specialization>,
    meta : Meta,
}

typedef FunctionDefinitionArguments : {
    arguments : Vector<TypedMember>,
    meta : Meta,
}

function named_token : (name : StringView, value : StringView, meta : Meta) -> NamedToken = {
    return {name, value, meta};
}

function identifier : (value : StringView, meta : Meta) -> NamedToken = {
   return named_token(sv("IDENTIFIER"), value, meta);
}

function concatinate : (start : StringView, end : StringView) -> Concatinate = {
    return {start, end};
}

function null : () -> Null = { return {}; }

function write_json_value : (stream : JSON_Stream&, _ : Null&) -> void = {
    stream.file:write(sv("null"));
}

function write_json_value : (stream : JSON_Stream&, text : Concatinate&) -> void = {
    stream:write("\"");
    stream:write_escaped(text.start);
    stream:write_escaped(text.end);
    stream:write("\"");
}

function write_json_value : (stream : JSON_Stream&, token : NamedToken&) -> void = {
    stream:write_token(token.name, token.meta, token.value);
}

function write_json_value : (stream : JSON_Stream&, fn : FunctionName&) -> void = {
    let meta : Meta = get_meta(fn.name);
    if fn.specialization:has_value() {
        meta.end_offset = fn.specialization:data().meta.end_offset;
    }

    if fn.specialization:has_value() {
        stream:write_tree(sv("function_name"), meta, stream:get_identifier(fn.name), fn.specialization:data());
    } else {
        stream:write_tree(sv("function_name"), meta, stream:get_identifier(fn.name), null());
    }
}

function write_json_value : (stream : JSON_Stream&, list : GenericDefinitionList&) -> void = {
    stream:write_tree(sv("generic_definition_list"), list.meta, list.generics);
}

function write_json_value : (stream : JSON_Stream&, list : SpecializationList&) -> void = {
    if list.specialization:length() == 0 {
        let null : Null = {};
        stream:write_json_value(&null);
    } else {
        stream:write_tree(sv("generic_specialization"), list.meta, list.specialization);
    }
}

function write_json_value : (stream : JSON_Stream&, args : FunctionDefinitionArguments&) -> void = {
    stream:write_tree(sv("function_arguments"), args.meta, args.arguments);
}

function [T] write_json_value : (stream : JSON_Stream&, opt : Optional<T>&) -> void = {
    if opt:has_value() {
        stream:write_json_value(&opt:data());
    } else {
        let null : Null = {}; // TODO: const &
        stream:write_json_value(&null);
    }
}

function write_json_value : (stream : JSON_Stream&, slice : ProgramSlice&) -> void = {
    let view : StringView = stream:get_sv(slice);
    stream:write_json_value(&view);
}

// END OF THE UNNECESSARY STUFF

function close : (stream : JSON_Stream&) -> void = {
    stream.file:close();
}

function write : (stream : JSON_Stream&, number : isize) -> void = {
    stream.file:write(number);
}

function write : (stream : JSON_Stream&, text : StringView) -> void = {
    stream.file:write(text);
}

function [@Len] write : (stream : JSON_Stream&, text : u8[@Len]&) -> void = {
    stream.file:write(sv(&text));
}

function get_meta : (slice : ProgramSlice) -> Meta = {
    return { slice.start_offset, slice.end_offset };
}

function get_sv : (stream : JSON_Stream&, slice : ProgramSlice) -> StringView = {
    return stream.program:slice_between(slice.start_offset, slice.end_offset);
}

function get_identifier : (stream : JSON_Stream&, value : ProgramSlice) -> NamedToken = {
    return identifier(stream:get_sv(value), get_meta(value));
}

function get_named_token : (stream : JSON_Stream&, name : StringView, value : ProgramSlice) -> NamedToken = {
    return named_token(name, stream:get_sv(value), get_meta(value));
}

function write_escaped : (stream : JSON_Stream&, text : StringView) -> void = {
    for char in text:get_iter() {
        if is_one_of(char, "\""[0], "\\"[0]) {
            stream.file:write_char("\\"[0]);
        }
        stream.file:write_char(char);
    }
}

function write_json_value : (stream : JSON_Stream&, text : StringView&) -> void = {
    // Escape everything that would break JSON
    stream.file:write_char("\""[0]);
    stream:write_escaped(text);
    stream.file:write_char("\""[0]);
}

function [T] write_json_value : (stream : JSON_Stream&, values : Vector<T>&) -> void = {
    // This is a bit misleading but it matches the lark implementation
    // The array ISN'T printed with brackets because lark collapses lists into the children
    let values_iter : VectorIter<T> = values:get_iter();
    if !values_iter:has_next() {
        return;
    }
    stream:write_json_value(&values_iter:__builtin_get_next());

    for value in values_iter {
        // TODO: references in for loops
        stream:write(",");
        stream:write_json_value(&value);
    }
}

function [T] write_key_value_pair : (stream : JSON_Stream&, key : StringView, value : T) -> void = {
    stream:write_json_value(&key);
    stream:write(":");
    stream:write_json_value(&value);
}

function [Others...] write_key_list_pair : (stream : JSON_Stream&, key : StringView, values : Others...) -> void = {
    stream:write_json_value(&key);
    stream:write(":");
    stream:write("[");
    write_inner_list(&stream, values...);
    stream:write("]");
}

function [Next] write_inner_list : (stream : JSON_Stream&, next : Next) -> void = {
    stream:write_json_value(&next);
}

function [Next, Others...] write_inner_list : (stream : JSON_Stream&, next : Next, others : Others...) -> void = {
    stream:write_json_value(&next);
    stream:write(",");

    // TODO: UFCS with parameter packs
    write_inner_list(&stream, others...);
}

function [T] write_token : (stream : JSON_Stream&, name : StringView, meta : Meta, value : T) -> void = {
    stream:write("{");
    stream:write_key_value_pair(sv("name"), name);
    stream:write(",");
    stream:write_key_value_pair(sv("value"), value);
    stream:write(",");
    stream:write_key_value_pair(sv("meta"), meta);
    stream:write("}");
}

function [Children...] write_tree : (stream : JSON_Stream&, name : StringView, meta : Meta, children : Children...) -> void = {
    stream:write("{");
    stream:write_key_value_pair(sv("name"), name);
    stream:write(",");
    // TODO: ufcs and parameter packs
    write_key_list_pair(&stream, sv("children"), children...);
    stream:write(",");
    stream:write_key_value_pair(sv("meta"), meta);
    stream:write("}");
}

function write_json_value : (stream : JSON_Stream&, meta : Meta&) -> void = {
    let start_position : FilePosition = stream.line_number_lookup:lookup_position(meta.start_offset);
    let end_position : FilePosition = stream.line_number_lookup:lookup_position(meta.end_offset);

    stream:write("{");
    stream:write_key_value_pair(sv("start"), start_position);
    stream:write(",");
    stream:write_key_value_pair(sv("end"), end_position);
    stream:write("}");
}

function write_json_value : (stream : JSON_Stream&, offset : FilePosition&) -> void = {
    stream:write("{");
    stream:write_key_value_pair(sv("line"), offset.line);
    stream:write(",");
    stream:write_key_value_pair(sv("column"), offset.column);
    stream:write("}");
}

function write_json_value : (stream : JSON_Stream&, number : isize&) -> void = {
    stream:write(number);
}


function write_json_value : (stream : JSON_Stream&, feature : TopLevelFeature&) -> void = {
    // These are some hacky vtables
    if feature.type == 0 {
        stream:write_json_value(&addr_to_ref<Typedef>(feature.pointer));
        return;
    }
    if feature.type == 1 {
        stream:write_json_value(&addr_to_ref<FunctionDefinition>(feature.pointer));
        return;
    }
    if feature.type == 2 {
        stream:write_json_value(&addr_to_ref<RequireOnce>(feature.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, constant : CompileTimeConstant&) -> void = {
    let name : StringView;
    if constant.is_identifier {
        name = sv("NUMERIC_GENERIC_IDENTIFIER");
    } else {
        name = sv("SIGNED_INT");
    }
    stream:write_tree(sv("compile_time_constant"), get_meta(constant.value), stream:get_named_token(name, constant.value));
}

function write_json_value : (stream : JSON_Stream&, fn : FunctionType&) -> void = {
    // TODO
}

function write_json_value : (stream : JSON_Stream&, array : ArrayType&) -> void = {
    let name : StringView;
    if array.is_heap_array {
        name = sv("heap_array_type");
        if array.size:length() == 0 {
            stream:write_tree(name, array.meta, array.base_type);
            return;
        }
    } else {
        name = sv("stack_array_type");
    }
    stream:write_tree(name, array.meta, array.base_type, array.size);
}

function write_json_value : (stream : JSON_Stream&, ref : RefType&) -> void = {
    stream:write_tree(sv("ref_type"), ref.meta, ref.type);
}

function write_json_value : (stream : JSON_Stream&, named_type : NamedType&) -> void = {
    stream:write_tree(sv("type_name"), named_type.meta, stream:get_identifier(named_type.name), named_type.specialization);
}

function write_json_value : (stream : JSON_Stream&, struct : StructType&) -> void = {
    stream:write_tree(sv("struct_type"), struct.meta, struct.members);
}

function write_json_value : (stream : JSON_Stream&, pack : PackType&) -> void = {
    // TODO: This lark emulation is nasty
    // TODO: atm we only support named types without a specialization
    runtime_assert(pack.type.type == 4);
    let named_type : NamedType& = &addr_to_ref<NamedType>(pack.type.pointer);
    runtime_assert(named_type.specialization.specialization:length() == 0);

    stream:write_token(sv("TYPE_PACK"), pack.meta, concatinate(stream:get_sv(named_type.name), sv("...")));
}

function write_json_value : (stream : JSON_Stream&, generic_def : GenericDefinition&) -> void = {
    if generic_def.is_packed {
        stream:write_token(sv("GENERIC_PACK_IDENTIFIER"), generic_def.meta, concatinate(stream:get_sv(generic_def.value), sv("...")));
    } else {
        stream:write_token(sv("GENERIC_IDENTIFIER"), generic_def.meta, generic_def.value);
    }
}

function write_json_value : (stream : JSON_Stream&, specialization_item : Specialization&) -> void = {
    if specialization_item.type == 0 {
        stream:write_json_value(&addr_to_ref<Type>(specialization_item.pointer));
        return;
    }
    if specialization_item.type == 1 {
        stream:write_json_value(&addr_to_ref<CompileTimeConstant>(specialization_item.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, type : Type&) -> void = {
    if type.type == 0 {
        stream:write_tree(sv("type"), type.meta, &addr_to_ref<FunctionType>(type.pointer));
        return;
    }
    if type.type == 1 {
        stream:write_tree(sv("type"), type.meta, &addr_to_ref<ArrayType>(type.pointer));
        return;
    }
    if type.type == 2 {
        stream:write_tree(sv("type"), type.meta, &addr_to_ref<RefType>(type.pointer));
        return;
    }
    // No 3 :-D
    if type.type == 4 {
        stream:write_tree(sv("type"), type.meta, &addr_to_ref<NamedType>(type.pointer));
        return;
    }
    if type.type == 5 {
        stream:write_tree(sv("type"), type.meta, &addr_to_ref<StructType>(type.pointer));
        return;
    }
    if type.type == 6 {
        // TODO: we handle this strangly to match lark
        stream:write_json_value(&addr_to_ref<PackType>(type.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, typed_member : TypedMember&) -> void = {
    // TODO: support const& (avoiding the temporary variable)
    let named_token : NamedToken = stream:get_identifier(typed_member.name);
    stream:write_json_value(&named_token);
    stream:write(",");
    stream:write_json_value(&typed_member.type);
}

function write_json_value : (stream : JSON_Stream&, fn_def : FunctionDefinition&) -> void = {
    // TODO: lark emulation here is painful
    if fn_def.generic_definitions.generics:length() == 0 {
        // TODO: parse functions with a specialization
        if fn_def.type == 0 {
            let fn_name : FunctionName = {fn_def.name, make_optional<SpecializationList>()};
            stream:write_tree(sv("specialized_named_function"), fn_def.meta, fn_name, fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
        if fn_def.type == 1 {
            stream:write_tree(sv("specialized_implicit_function"), fn_def.meta, stream:get_named_token(sv("IMPLICIT_FUNCTIONS"), fn_def.name), null(), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
        if fn_def.type == 2 {
            stream:write_tree(sv("specialized_operator_function"), fn_def.meta, stream:get_named_token(sv("OPERATOR"), fn_def.name), null(), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
        if fn_def.type == 3 {
            stream:write_tree(sv("foreign_function"), fn_def.meta, stream:get_identifier(fn_def.name), fn_def.args, fn_def.return_type);
            return;
        }
        if fn_def.type == 4 {
            stream:write_tree(sv("specialized_assignment_function"), fn_def.meta, stream:get_named_token(sv("ASSIGNMENT_OPERATOR"), fn_def.name), null(), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
    } else {
        if fn_def.type == 0 {
            stream:write_tree(sv("generic_named_function"), fn_def.meta, fn_def.generic_definitions, stream:get_identifier(fn_def.name), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
        if fn_def.type == 1 {
            stream:write_tree(sv("generic_implicit_function"), fn_def.meta, fn_def.generic_definitions, stream:get_named_token(sv("IMPLICIT_FUNCTIONS"), fn_def.name), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
        if fn_def.type == 2 {
            stream:write_tree(sv("generic_operator_function"), fn_def.meta, fn_def.generic_definitions, stream:get_named_token(sv("OPERATOR"), fn_def.name), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
        if fn_def.type == 4 {
            stream:write_tree(sv("generic_assignment_function"), fn_def.meta, fn_def.generic_definitions, stream:get_named_token(sv("ASSIGNMENT_OPERATOR"), fn_def.name), fn_def.args, fn_def.return_type, fn_def.scope);
            return;
        }
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, require : RequireOnce&) -> void = {
    stream:write_tree(sv("require_once"), require.meta, stream:get_named_token(sv("ESCAPED_STRING"), require.file_path));
}

function write_json_value : (stream : JSON_Stream&, td : Typedef&) -> void = {
    if td.specialization.specialization:length() == 0 {
        // Ewww
        if td.generic_definitions.generics:length() == 0 {
            stream:write_tree(sv("generic_typedef"), td.meta, null(), stream:get_identifier(td.name), td.type);
        } else {
            stream:write_tree(sv("generic_typedef"), td.meta, td.generic_definitions, stream:get_identifier(td.name), td.type);
        }
    } else {
        stream:write_tree(sv("specialized_typedef"), td.meta, stream:get_identifier(td.name), td.specialization, td.type);
    }
}

function write_json_value : (stream : JSON_Stream&, scope : Scope&) -> void = {
    stream:write_tree(sv("scope"), scope.meta, scope.lines);
}

function write_json_value : (stream : JSON_Stream&, line : Line&) -> void = {
    if line.type == 0 {
        stream:write_json_value(&addr_to_ref<Expression>(line.pointer));
        return;
    }
    if line.type == 1 {
        stream:write_json_value(&addr_to_ref<Assignment>(line.pointer));
        return;
    }
    if line.type == 2 {
        stream:write_json_value(&addr_to_ref<Return>(line.pointer));
        return;
    }
    if line.type == 3 {
        stream:write_json_value(&addr_to_ref<If>(line.pointer));
        return;
    }
    if line.type == 4 {
        stream:write_json_value(&addr_to_ref<While>(line.pointer));
        return;
    }
    if line.type == 5 {
        stream:write_json_value(&addr_to_ref<For>(line.pointer));
        return;
    }
    if line.type == 6 {
        stream:write_json_value(&addr_to_ref<Scope>(line.pointer));
        return;
    }
    if line.type == 7 {
        stream:write_json_value(&addr_to_ref<Declaration>(line.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, node : If&) -> void = {
    stream:write_tree(sv("if_statement"), node.meta, node.condition, node.if_scope, node.else_scope);
}

function write_json_value : (stream : JSON_Stream&, node : While&) -> void = {
    stream:write_tree(sv("while_statement"), node.meta, node.condition, node.scope);
}

function write_json_value : (stream : JSON_Stream&, node : For&) -> void = {
    stream:write_tree(sv("for_statement"), node.meta, stream:get_identifier(node.variable), node.iterator_expression, node.scope);
}

function write_json_value : (stream : JSON_Stream&, node : Return&) -> void = {
    stream:write_tree(sv("return_statement"), node.meta, node.expression);
}

function write_json_value : (stream : JSON_Stream&, node : Assignment&) -> void = {
    stream:write_tree(sv("assignment"), node.meta, node.lhs, stream:get_named_token(sv("ASSIGNMENT_OPERATOR"), node.operator), node.rhs);
}

function write_json_value : (stream : JSON_Stream&, node : Declaration&) -> void = {
    if node.is_const {
        runtime_assert(node.expression:has_value());
        stream:write_tree(sv("const_declaration"), node.meta, stream:get_identifier(node.variable), node.type, node.expression:data());
    } else {
        stream:write_tree(sv("variable_declaration"), node.meta, stream:get_identifier(node.variable), node.type, node.expression);
    }
}

function write_json_value : (stream : JSON_Stream&, node : Expression&) -> void = {
    if node.type == 0 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<FunctionCall>(node.pointer));
        return;
    }
    if node.type == 1 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<Constant>(node.pointer));
        return;
    }
    if node.type == 2 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<NamedInitializerList>(node.pointer));
        return;
    }
    if node.type == 3 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<UnnamedInitializerList>(node.pointer));
        return;
    }
    if node.type == 4 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<VariableAccess>(node.pointer));
        return;
    }
    if node.type == 5 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<ArrayIndexAccess>(node.pointer));
        return;
    }
    if node.type == 6 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<StructIndexAccess>(node.pointer));
        return;
    }
    if node.type == 7 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<UFCS_Call>(node.pointer));
        return;
    }
    if node.type == 8 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<Operator>(node.pointer));
        return;
    }
    if node.type == 9 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<UnaryOperator>(node.pointer));
        return;
    }
    if node.type == 10 {
        stream:write_json_value(&addr_to_ref<PackExpansion>(node.pointer));
        return;
    }
    if node.type == 11 {
        stream:write_tree(sv("expression"), node.meta, &addr_to_ref<Borrow>(node.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, node : NamedExpression&) -> void = {
    // Hacky for lark.. This would normally be borked JSON
    let identifier : NamedToken = stream:get_identifier(node.name); // TODO: const ref
    stream:write_json_value(&identifier);
    stream:write(",");
    stream:write_json_value(&node.expression);
}

function write_json_value : (stream : JSON_Stream&, node : Operator&) -> void = {
    stream:write_tree(sv("operator_use"), node.meta, node.lhs, stream:get_named_token(sv("OPERATOR"), node.name), node.rhs);
}

function write_json_value : (stream : JSON_Stream&, node : UnaryOperator&) -> void = {
    stream:write_tree(sv("unary_operator_use"), node.meta, stream:get_named_token(sv("UNARY_OPERATOR_TOKEN"), node.name), node.rhs);
}

function write_json_value : (stream : JSON_Stream&, node : Borrow&) -> void = {
    if node.is_const {
        stream:write_tree(sv("const_borrow_operator_use"), node.meta, node.expression);
    } else {
        stream:write_tree(sv("borrow_operator_use"), node.meta, node.expression);
    }
}

function write_json_value : (stream : JSON_Stream&, node : FunctionCall&) -> void = {
    let function_name : FunctionName = {node.name, make_optional(node.specialization)};

    // Emulate the strange unpacking behaviour
    if node.arguments:length() == 0 {
        stream:write_tree(sv("function_call"), node.meta, function_name, null());
        return;
    }
    if node.arguments:get(node.arguments:length() - 1).type != 10 {
        // The last argument is NOT a parameter pack expansion
        stream:write_tree(sv("function_call"), node.meta, function_name, node.arguments, null());
        return;
    }
    // The last argument IS a parameter pack expansion
    stream:write_tree(sv("function_call"), node.meta, function_name, node.arguments);
}

function write_json_value : (stream : JSON_Stream&, node : UFCS_Call&) -> void = {
    let function_name : FunctionName = {node.function_name, make_optional(node.specialization)};

    // Emulate the strange unpacking behaviour
    if node.args:length() == 0 {
        stream:write_tree(sv("ufcs_call"), node.meta, node.expression, function_name, null());
        return;
    }
    if node.args:get(node.args:length() - 1).type != 10 {
        // The last argument is NOT a parameter pack expansion
        stream:write_tree(sv("ufcs_call"), node.meta, node.expression, function_name, node.args, null());
        return;
    }
    // The last argument IS a parameter pack expansion
    stream:write_tree(sv("ufcs_call"), node.meta, node.expression, function_name, node.args);
}

function write_json_value : (stream : JSON_Stream&, node : PackExpansion&) -> void = {
    // TODO: support more types of pack
    runtime_assert(node.expression.type == 4 /* variable */);
    let variable : VariableAccess& = &addr_to_ref<VariableAccess>(node.expression.pointer);
    stream:write_token(sv("ACCESSED_PACK_NAME"), node.meta, concatinate(stream:get_sv(variable.name), sv("...")));
}

function write_json_value : (stream : JSON_Stream&, node : Constant&) -> void = {
    // TODO: JSON actually supports these types natively
    //    no need to do this in the compiler (other than lark compat)
    if node.type == 0 {
        stream:write_token(sv("ESCAPED_STRING"), get_meta(node.value), node.value);
        return;
    }
    if node.type == 1 {
        // TODO: floats
        runtime_assert(false);
    }
    if node.type == 2 {
        // TODO: support hex
        stream:write_token(sv("SIGNED_INT"), get_meta(node.value), node.value);
        return;
    }
    if node.type == 3 {
        stream:write_token(sv("BOOL_CONSTANT"), get_meta(node.value), node.value);
        return;
    }
    if node.type == 4 {
        stream:write_token(sv("NUMERIC_GENERIC_IDENTIFIER"), get_meta(node.value), node.value);
        return;
    }
    if node.type == 5 {
        stream:write_token(sv("UNSIGNED_HEX_CONSTANT"), get_meta(node.value), node.value);
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream&, node : NamedInitializerList&) -> void = {
    stream:write_tree(sv("struct_initializer_with_names"), node.meta, node.args);
}

function write_json_value : (stream : JSON_Stream&, node : UnnamedInitializerList&) -> void = {
    stream:write_tree(sv("struct_initializer_without_names"), node.meta, node.args);
}

function write_json_value : (stream : JSON_Stream&, node : VariableAccess&) -> void = {
    stream:write_tree(sv("accessed_variable_name"), get_meta(node.name), stream:get_identifier(node.name));
}

function write_json_value : (stream : JSON_Stream&, node : ArrayIndexAccess&) -> void = {
    stream:write_tree(sv("array_index_access"), node.meta, node.expression, node.indicies);
}

function write_json_value : (stream : JSON_Stream&, node : StructIndexAccess&) -> void = {
    stream:write_tree(sv("struct_member_access"), node.meta, node.expression, stream:get_identifier(node.member));
}

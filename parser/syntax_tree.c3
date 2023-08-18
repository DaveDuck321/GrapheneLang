@require_once "std/vector.c3"
@require_once "std/string.c3"

typedef ProgramSlice : {
    start : isize,
    end : isize,
}

typedef TopLevelFeature : {
    type : int, // 0: typedef, 1: function, 2: require once
    pointer : iptr, // TODO: the indirection here is unneeded... a union would be better
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
typedef PackType : { type : Type }

typedef GenericDefinition : {
    // TODO: variant
    type : int, // 0: type, 1: numeric generic
    value : ProgramSlice,
    is_packed : bool,
}

typedef Specialization : {
    // TODO: variant
    type : int, // 0: type, 1: compile time constant
    pointer : iptr,
}

typedef Type : {
    // TODO: variant
    type: int, // 0: function, 1: array, 2: ref, 4: named type, 5: struct type, 6: parameter pack
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
    type : int, // 0: expression, 1: assignment, 2: return, 3: if, 4: while, 5: for, 6: scope, 7: declaration
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
    expression : Optional<Expression>,
}

typedef Expression : {
    // TODO: this is a poor man's variant
    // TODO: use an enum here

    // 0: function call, 1: constant, 2: named init list, 3: unnamed init list, 4: variable, 5: array index access, 6: struct index access, 7: ufcs, 8: operator, 9: unary operator, 10: pack expansion
    type : int,
    pointer : iptr,
}

typedef NamedExpression : {
    name : ProgramSlice,
    expression : Expression,
}

typedef Operator : {
    // TODO: this can probably be unified with the function call type
    name : StringView,
    lhs : Expression,
    rhs : Expression,
}

typedef UnaryOperator : {
    name : ProgramSlice,
    rhs : Expression,
}

typedef FunctionCall : {
    name : ProgramSlice,
    specialization : Vector<Specialization>,
    arguments : Vector<Expression>,
}

typedef UFCS_Call : {
    expression : Expression,
    function_name : ProgramSlice,
    specialization : Vector<Specialization>,
    args : Vector<Expression>,
}

typedef PackExpansion : {
    expression : Expression,
}

typedef Constant : {
    // TODO: enum
    type : int, // 0: String, 1: float, 2: int, 3: bool, 4: generic identifier
    value : ProgramSlice,
}

typedef NamedInitializerList : {
    args : Vector<NamedExpression>,
}

typedef UnnamedInitializerList : {
    args : Vector<Expression>
}

typedef VariableAccess : {
    name : ProgramSlice
}

typedef ArrayIndexAccess : {
    expression : Expression,
    indicies: Vector<Expression>,
}

typedef StructIndexAccess : {
    expression : Expression,
    member : ProgramSlice,
}


// Output
typedef JSON_Stream : {
    // TODO: something smarter than this
    stringify_buffer : u8[1024],
    data : Vector<u8>,
    file : File,
}

function write : (stream : JSON_Stream&, text : StringView) -> void = {
    stream.data:extend_back(text);
}

function get_empty_meta : () -> StringView = {
    return sv("{\"start\":{\"line\":0, \"column\":0}, \"end\":{\"line\":0, \"column\":0}}");
}

function [T] output_key_value_pair : (stream : JSON_Stream&, key : StringView, value : T) -> void = {
    stream:write("\"");
    stream:write(key);
    stream:write("\" : ");
    stream:output_value(value);
}

function [T] output_token : (stream : JSON_Stream&, name : StringView, value : T) -> void = {
    // TODO: support meta
    let value : StringView = stringify(stream.stringify_buffer, value);

    stream:write("{");
    stream:output_key_value_pair("name", name);
    stream:write(",");
    stream:output_key_value_pair("value", value);
    stream:write(",");
    stream:output_key_value_pair("meta", get_empty_meta());
    stream:write("}")
}

function [Children...] output_tree : (stream : JSON_Stream&, name : StringView, children : Children...) -> void = {
    // TODO: support meta
    let value : StringView = stringify(stream.stringify_buffer, value);

    stream:write("{");
    stream:output_key_value_pair("name", name);
    stream:write(",");
    stream:output_key_value_pair("children", value); // TODO: here?
    stream:write(",");
    stream:output_key_value_pair("meta", get_empty_meta());
    stream:write("}")
}

function [Next] stream_inner_list : (stream : JSON_Stream&, next : Next) -> void = {
    stream:stream(next);
}

function [Next, Others...] stream_inner_list : (stream : JSON_Stream&, next : Next, others : Others...) -> void = {
    stream:stream(next, ", ");
    stream:stream_inner_list(others);
}

function [Children...] stream_tree : (stream : JSON_Stream&, name : StringView, children : Children...) -> void = {
    stream:stream("{\"name\":\"", name, "\", \"children\":[", stream:stream_inner_list(children...), "], ", get_empty_meta(), "}");
}

// TODO: eventually these should print to a file NOT stdout
function json_stringify : (stream : JSON_Stream&, feature : TopLevelFeature&) -> void = {
    // These are some hacky vtables
    if feature.type == 0 {
        return stream:json_stringify(&addr_to_ref<Typedef>(feature.pointer));
    }
    if feature.type == 1 {
        return stream:json_stringify(&addr_to_ref<FunctionDefinition>(feature.pointer));
    }
    if feature.type == 2 {
        return stream:json_stringify(&addr_to_ref<RequireOnce>(feature.pointer));
    }
    runtime_assert(false);
}

function json_stringify : (stream : JSON_Stream&, constant : CompileTimeConstant&) -> void = {
    if constant.is_identifier {
        return stream:stream_token("NUMERIC_GENERIC_IDENTIFIER", constant.value);
    } else {
        return stream:stream_token("SIGNED_INT", constant.value);
    }
    runtime_assert(false);
}

function json_stringify : (stream : JSON_Stream&, fn : FunctionType&) -> void = {
    // TODO
}

function json_stringify : (stream : JSON_Stream&, array : ArrayType&) -> void = {
    let name : StringView;
    if array.is_heap_array {
        name = sv("heap_array_type");
    } else {
        name = sv("stack_array_type");
    }

}

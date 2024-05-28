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
typedef ArrayType : { base_type : Type, size : Vector<CompileTimeConstant>, is_heap_array : bool, is_mut : bool, meta : Meta}
typedef RefType : { type : Type, is_mut : bool, meta : Meta }
typedef NamedType : { name : ProgramSlice, specialization : Vector<Specialization>, meta : Meta}
typedef StructType : { members : Vector<TypedMember>, meta : Meta}
typedef PackType : { type : Type, meta : Meta }

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
}

typedef Type : {
    // TODO: variant
    type: int, // 0: function, 1: array, 2: ref, 4: named type, 5: struct type, 6: parameter pack
    pointer : iptr,
    meta : Meta,
}

typedef TypedMember : {
    name : ProgramSlice,
    type : Type,
}

typedef FunctionDefinition : {
    // TODO: enum
    type : int, // 0: graphene, 1: implicit, 2: operator, 3: foreign, 4: assignment
    generic_definitions : Vector<GenericDefinition>,
    name : ProgramSlice,
    specialization : Vector<Specialization>,
    args : Vector<TypedMember>,
    return_type : Type,
    scope : Scope,
    meta : Meta,
}

typedef RequireOnce : {
    path : ProgramSlice,
    meta : Meta,
}

typedef Typedef : {
    name : ProgramSlice,
    generic_definitions : Vector<GenericDefinition>,
    specialization : Vector<Specialization>,
    alias : Type,
    meta : Meta,
}

typedef Scope : { lines : Vector<Line>, meta : Meta }
typedef Line : {
    // TODO: enum
    type : int, // 0: expression, 1: assignment, 2: return, 3: if, 4: while, 5: for, 6: scope, 7: declaration, 8: continue, 9: break
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
    iterator : Expression,
    scope : Scope,
    meta : Meta,
}

typedef Return : { expression : Optional<Expression>, meta : Meta }

typedef Continue : { meta : Meta }

typedef Break : { meta : Meta }

typedef Assignment : {
    lhs : Expression,
    operator : ProgramSlice,
    rhs : Expression,
    meta : Meta,
}

typedef VariableDeclaration : {
    is_mut : bool,
    variable : ProgramSlice,
    type : Type,
    expression : Optional<Expression>,
    meta : Meta,
}

typedef Expression : {
    // TODO: this is a poor man's variant
    // TODO: use an enum here

    // 0: function call, 1: constant, 2: named init list, 3: unnamed init list,
    // 4: variable, 5: array index access, 6: struct index access, 7: ufcs, 8:
    // operator, 9: unary operator, 10: pack expansion, 11: borrow, 12: logical
    // operator
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

typedef LogicalOperator : {
    name : ProgramSlice,
    lhs : Expression,
    rhs : Expression,
    meta : Meta,
}

typedef Borrow : {
    is_mut : bool,
    expression : Expression,
    meta : Meta,
}

typedef FunctionCall : {
    name : ProgramSlice,
    specialization : Vector<Specialization>,
    args : Vector<Expression>,
    meta : Meta,
}

typedef UFCS_Call : {
    expression : Expression,
    fn_name : ProgramSlice,
    specialization : Vector<Specialization>,
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
    indexes: Vector<Expression>,
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

typedef JSON_Object : {
    is_open : bool,
    length : int,
    stream : JSON_Stream mut&,
}

function new_object : (stream : JSON_Stream mut&) -> JSON_Object = {
    stream:write("{");
    return {true, 0, &mut stream};
}

function [@Len] add_meta : (object : JSON_Object mut&, type : u8[@Len]&, meta : Meta) -> JSON_Object mut& = {
    return &mut object:add_meta(sv(&type), meta);
}

function add_meta : (object : JSON_Object mut&, type : StringView, meta : Meta) -> JSON_Object mut& = {
    object:add("__type__", type);
    object:add("meta", meta);
    return &mut object;
}

function [T, @Len] add : (object : JSON_Object mut&, key : u8[@Len]&, value : T) -> JSON_Object mut& = {
    return &mut object:add(sv(&key), value);
}

function [T] add : (object : JSON_Object mut&, key : StringView, value : T) -> JSON_Object mut& = {
    if object.length != 0 {
        object.stream:write(",");
    }
    object.length += 1;
    object.stream:write_key_value_pair(key, value);
    return &mut object;
}

function end : (object : JSON_Object mut&) -> void = {
    runtime_assert(object.is_open);
    object.is_open = false;
    object.stream:write("}");
}

typedef Null : {}
typedef RawValue : { value : ProgramSlice }

function null : () -> Null = { return {}; }
function raw : (value : ProgramSlice) -> RawValue = {
    return { value };
}

function close : (stream : JSON_Stream mut&) -> void = {
    stream.file:close();
}

function get_meta : (slice : ProgramSlice) -> Meta = {
    return { slice.start_offset, slice.end_offset };
}

function get_sv : (stream : JSON_Stream mut&, slice : ProgramSlice) -> StringView = {
    let view : View<u8> = slice_between(&stream.program.buffer, slice.start_offset, slice.end_offset);
    return {view};
}

function [@Len] write : (stream : JSON_Stream mut&, text : u8[@Len]&) -> void = {
    stream.file:write(sv(&text));
}

function write : (stream : JSON_Stream mut&, text : StringView) -> void = {
    stream.file:write(text);
}

function [T] write_key_value_pair : (stream : JSON_Stream mut&, key : StringView, value : T) -> void = {
    stream:write_json_value(&key);
    stream:write(":");
    stream:write_json_value(&value);
}

function write_json_value : (stream : JSON_Stream mut&, _ : Null&) -> void = {
    stream:write(sv("null"));
}

function write_json_value : (stream : JSON_Stream mut&, value : RawValue&) -> void = {
    stream:write(stream:get_sv(value.value));
}

function [T] write_json_value : (stream : JSON_Stream mut&, opt : Optional<T>&) -> void = {
    if opt:has_value() {
        stream:write_json_value(&opt:data());
    } else {
        let null : Null = {}; // TODO: let &
        stream:write_json_value(&null);
    }
}

function write_json_value : (stream : JSON_Stream mut&, slice : ProgramSlice&) -> void = {
    let view : StringView = stream:get_sv(slice);
    stream:write_json_value(&view);
}

function write_escaped : (stream : JSON_Stream mut&, text : StringView) -> void = {
    for char in text:get_iter() {
        if is_one_of(char, "\""[0], "\\"[0]) {
            stream.file:write_char("\\"[0]);
        }
        stream.file:write_char(char);
    }
}

function write_json_value : (stream : JSON_Stream mut&, text : StringView&) -> void = {
    // Escape everything that would break JSON
    stream:write("\"");
    stream:write_escaped(text);
    stream:write("\"");
}

function write_json_value : (stream : JSON_Stream mut&, number : isize&) -> void = {
    stream.file:write(number);
}

function write_json_value : (stream : JSON_Stream mut&, value : bool&) -> void = {
    stream.file:write(value);
}

function [T] write_json_value : (stream : JSON_Stream mut&, values : Vector<T>&) -> void = {
    stream:write("[");
    mut values_iter : ContiguousContainerIter<Vector<T>&, T&> = values:get_citer();
    if !values_iter:has_next() {
        stream:write("]");
        return;
    }
    stream:write_json_value(&values_iter:__builtin_get_next());

    for value in values_iter {
        stream:write(",");
        stream:write_json_value(&value);
    }
    stream:write("]");
}

function write_json_value : (stream : JSON_Stream mut&, meta : Meta&) -> void = {
    let start_position : FilePosition = stream.line_number_lookup:lookup_position(meta.start_offset);
    let end_position : FilePosition = stream.line_number_lookup:lookup_position(meta.end_offset);

    mut object : JSON_Object = stream:new_object();
    object
        :add("__type__", sv("Meta"))
        :add("start", start_position)
        :add("end", end_position)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, offset : FilePosition&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add("__type__", sv("FilePosition"))
        :add("line", offset.line)
        :add("column", offset.column)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, feature : TopLevelFeature&) -> void = {
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

function write_json_value : (stream : JSON_Stream mut&, constant : CompileTimeConstant&) -> void = {
    mut object : JSON_Object = stream:new_object();
    if constant.is_identifier {
        object:add_meta("NumericGenericIdentifier", get_meta(constant.value));
        object:add("value", constant.value);
    } else {
        object:add_meta("NumericIdentifier", get_meta(constant.value));
        object:add("value", raw(constant.value));
    }
    object:end();
}

function write_json_value : (stream : JSON_Stream mut&, fn : FunctionType&) -> void = {
    // TODO
}

function write_json_value : (stream : JSON_Stream mut&, array : ArrayType&) -> void = {
    mut object : JSON_Object = stream:new_object();
    if array.is_heap_array {
        object:add_meta("HeapArrayType", array.meta);
        object:add("is_mut", array.is_mut);
    } else {
        object:add_meta("StackArrayType", array.meta);
    }
    object
        :add("base_type", array.base_type)
        :add("size", array.size)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : RefType&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("ReferenceType", node.meta)
        :add("value_type", node.type)
        :add("is_mut", node.is_mut)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : NamedType&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("NamedType", node.meta)
        :add("name", node.name)
        :add("specialization", node.specialization)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : StructType&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("StructType", node.meta)
        :add("members", node.members)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : PackType&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("PackType", node.meta)
        :add("type_", node.type)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : GenericDefinition&) -> void = {
    mut object : JSON_Object = stream:new_object();
    mut type : StringView;
    if node.type == 0 {
        type = sv("TypeGenericDefinition");
    } else {
        runtime_assert(node.type == 1);
        type = sv("NumericGenericDefinition");
    }

    object
        :add_meta(type, node.meta)
        :add("name", node.value)
        :add("is_packed", node.is_packed)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, specialization_item : Specialization&) -> void = {
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

function write_json_value : (stream : JSON_Stream mut&, type : Type&) -> void = {
    if type.type == 0 {
        stream:write_json_value(&addr_to_ref<FunctionType>(type.pointer));
        return;
    }
    if type.type == 1 {
        stream:write_json_value(&addr_to_ref<ArrayType>(type.pointer));
        return;
    }
    if type.type == 2 {
        stream:write_json_value(&addr_to_ref<RefType>(type.pointer));
        return;
    }
    // No 3 :-D
    if type.type == 4 {
        stream:write_json_value(&addr_to_ref<NamedType>(type.pointer));
        return;
    }
    if type.type == 5 {
        stream:write_json_value(&addr_to_ref<StructType>(type.pointer));
        return;
    }
    if type.type == 6 {
        stream:write_json_value(&addr_to_ref<PackType>(type.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream mut&, node : TypedMember&) -> void = {
    stream:write("[");
    stream:write_json_value(&node.name);
    stream:write(",");
    stream:write_json_value(&node.type);
    stream:write("]");
}

function write_json_value : (stream : JSON_Stream mut&, fn_def : FunctionDefinition&) -> void = {
    runtime_assert(0 <= fn_def.type and fn_def.type <= 4);

    mut object : JSON_Object = stream:new_object();
    object
        :add("name", fn_def.name)
        :add("args", fn_def.args)
        :add("return_", fn_def.return_type);

    if fn_def.type == 3 {
        object:add_meta("ForeignFunction", fn_def.meta);
        object:end();
        return;
    }

    if fn_def.type == 0 {
        object:add_meta("GrapheneFunction", fn_def.meta);
    }
    if fn_def.type == 1 {
        object:add_meta("ImplicitFunction", fn_def.meta);
    }
    if fn_def.type == 2 {
        object:add_meta("OperatorFunction", fn_def.meta);
    }
    if fn_def.type == 4 {
        object:add_meta("AssignmentFunction", fn_def.meta);
    }

    object
        :add("generic_definitions", fn_def.generic_definitions)
        :add("specialization", fn_def.specialization)
        :add("scope", fn_def.scope)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : RequireOnce&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("RequireOnce", node.meta)
        :add("path", node.path)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Typedef&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Typedef", node.meta)
        :add("name", node.name)
        :add("generic_definitions", node.generic_definitions)
        :add("specialization", node.specialization)
        :add("alias", node.alias)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Scope&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Scope", node.meta)
        :add("lines", node.lines)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, line : Line&) -> void = {
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
        stream:write_json_value(&addr_to_ref<VariableDeclaration>(line.pointer));
        return;
    }
    if line.type == 8 {
        stream:write_json_value(&addr_to_ref<Continue>(line.pointer));
        return;
    }
    if line.type == 9 {
        stream:write_json_value(&addr_to_ref<Break>(line.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream mut&, node : If&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("If", node.meta)
        :add("condition", node.condition)
        :add("if_scope", node.if_scope)
        :add("else_scope", node.else_scope)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : While&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("While", node.meta)
        :add("condition", node.condition)
        :add("scope", node.scope)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : For&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("For", node.meta)
        :add("variable", node.variable)
        :add("iterator", node.iterator)
        :add("scope", node.scope)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Return&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Return", node.meta)
        :add("expression", node.expression)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Continue&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Continue", node.meta)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Break&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Break", node.meta)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Assignment&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Assignment", node.meta)
        :add("lhs", node.lhs)
        :add("rhs", node.rhs)
        :add("operator", node.operator)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : VariableDeclaration&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("VariableDeclaration", node.meta)
        :add("is_mut", node.is_mut)
        :add("variable", node.variable)
        :add("type_", node.type)
        :add("expression", node.expression)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Expression&) -> void = {
    if node.type == 0 {
        stream:write_json_value(&addr_to_ref<FunctionCall>(node.pointer));
        return;
    }
    if node.type == 1 {
        stream:write_json_value(&addr_to_ref<Constant>(node.pointer));
        return;
    }
    if node.type == 2 {
        stream:write_json_value(&addr_to_ref<NamedInitializerList>(node.pointer));
        return;
    }
    if node.type == 3 {
        stream:write_json_value(&addr_to_ref<UnnamedInitializerList>(node.pointer));
        return;
    }
    if node.type == 4 {
        stream:write_json_value(&addr_to_ref<VariableAccess>(node.pointer));
        return;
    }
    if node.type == 5 {
        stream:write_json_value(&addr_to_ref<ArrayIndexAccess>(node.pointer));
        return;
    }
    if node.type == 6 {
        stream:write_json_value(&addr_to_ref<StructIndexAccess>(node.pointer));
        return;
    }
    if node.type == 7 {
        stream:write_json_value(&addr_to_ref<UFCS_Call>(node.pointer));
        return;
    }
    if node.type == 8 {
        stream:write_json_value(&addr_to_ref<Operator>(node.pointer));
        return;
    }
    if node.type == 9 {
        stream:write_json_value(&addr_to_ref<UnaryOperator>(node.pointer));
        return;
    }
    if node.type == 10 {
        stream:write_json_value(&addr_to_ref<PackExpansion>(node.pointer));
        return;
    }
    if node.type == 11 {
        stream:write_json_value(&addr_to_ref<Borrow>(node.pointer));
        return;
    }
    if node.type == 12 {
        stream:write_json_value(&addr_to_ref<LogicalOperator>(node.pointer));
        return;
    }
    runtime_assert(false);
}

function write_json_value : (stream : JSON_Stream mut&, node : NamedExpression&) -> void = {
    stream:write("[");
    stream:write_json_value(&node.name);
    stream:write(",");
    stream:write_json_value(&node.expression);
    stream:write("]");
}

function write_json_value : (stream : JSON_Stream mut&, node : Operator&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("OperatorUse", node.meta)
        :add("name", node.name)
        :add("lhs", node.lhs)
        :add("rhs", node.rhs)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : UnaryOperator&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("UnaryOperatorUse", node.meta)
        :add("name", node.name)
        :add("rhs", node.rhs)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : LogicalOperator&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("LogicalOperatorUse", node.meta)
        :add("name", node.name)
        :add("lhs", node.lhs)
        :add("rhs", node.rhs)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Borrow&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("Borrow", node.meta)
        :add("is_mut", node.is_mut)
        :add("expression", node.expression)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : FunctionCall&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("FunctionCall", node.meta)
        :add("name", node.name)
        :add("specialization", node.specialization)
        :add("args", node.args)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : UFCS_Call&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("UFCS_Call", node.meta)
        :add("expression", node.expression)
        :add("name", node.fn_name)
        :add("specialization", node.specialization)
        :add("args", node.args)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : PackExpansion&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("PackExpansion", node.meta)
        :add("expression", node.expression)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : Constant&) -> void = {
    runtime_assert(0 <= node.type and node.type <= 5);

    mut object : JSON_Object = stream:new_object();
    let meta : Meta = get_meta(node.value);

    // JSON supports some types natively, no need to export all of them as strings
    if node.type == 0 {
        object:add_meta("StringConstant", meta);
        object:add("value", node.value);
    }
    if node.type == 1 {
        object:add_meta("FloatConstant", meta);
        object:add("value", node.value);
    }
    if node.type == 2 {
        object:add_meta("IntConstant", meta);
        object:add("value", raw(node.value));
    }
    if node.type == 3 {
        object:add_meta("BoolConstant", meta);
        object:add("value", raw(node.value));
    }
    if node.type == 4 {
        object:add_meta("GenericIdentifierConstant", meta);
        object:add("value", node.value);
    }
    if node.type == 5 {
        object:add_meta("HexConstant", meta);
        object:add("value", node.value);
    }
    object:end();
}

function write_json_value : (stream : JSON_Stream mut&, node : NamedInitializerList&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("NamedInitializerList", node.meta)
        :add("args", node.args)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : UnnamedInitializerList&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("UnnamedInitializerList", node.meta)
        :add("args", node.args)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : VariableAccess&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("VariableAccess", get_meta(node.name))
        :add("name", node.name)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : ArrayIndexAccess&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("ArrayIndexAccess", node.meta)
        :add("expression", node.expression)
        :add("indexes", node.indexes)
        :end();
}

function write_json_value : (stream : JSON_Stream mut&, node : StructIndexAccess&) -> void = {
    mut object : JSON_Object = stream:new_object();
    object
        :add_meta("StructIndexAccess", node.meta)
        :add("expression", node.expression)
        :add("member", node.member)
        :end();
}

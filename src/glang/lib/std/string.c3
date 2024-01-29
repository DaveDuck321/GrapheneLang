@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "iterators.c3"
@require_once "util.c3"
@require_once "wrappers.c3"
@require_once "span.c3"

typedef CString : { data : u8[&] }
typedef StringView     : { buffer : View<u8> }
typedef UTF32Codepoint : { codepoint : u8 /* TODO: unicode support */ }

function decode_utf8 : (char : u8) -> UTF32Codepoint = {
    // TODO: this just supports ascii atm
    return {char};
}

// StringView
function [@Len] sv : (string : u8[@Len]&) -> StringView = {
    return { make<View<u8>>(&string) };
}

function make<StringView> : (buffer : View<u8>) -> StringView = {
    return { buffer };
}

function make<StringView> : (cstr : CString) -> StringView = {
    return { make<View<u8>>(&cstr.data, cstr:length()) };
}

function length : (sv : StringView&) -> isize = {
    return sv.buffer:length();
}

// TODO: return `UTF32Codepoint` here
function get_iter : (sv : StringView&) -> ContiguousContainerIter<View<u8>&, u8&> = {
    return sv.buffer:get_iter();
}

@operator == : (lhs : StringView, rhs : StringView) -> bool = {
    return lhs.buffer == rhs.buffer;
}

function slice_to : (sv : StringView&, end : isize) -> StringView = {
    return { sv.buffer:slice_to(end) };
}

function slice_from : (sv : StringView&, start : isize) -> StringView = {
    return { sv.buffer:slice_from(start) };
}

function slice_between : (sv : StringView&, start : isize, end : isize) -> StringView = {
    return { sv.buffer:slice_between(start, end) };
}

// CString
function length : (str: CString&) -> isize = {
    mut len : isize = 0;
    while str.data[len] != char("\0") {
        len = len + 1;
    }
    return len;
}

function [@Len] make<CString> : (literal : u8[@Len]&) -> CString = {
    runtime_assert(literal[@Len - 1] == char("\0"));
    return {&literal};
}


// String utilities
function ascii_is_white_space : (char : u8) -> bool = {
    // TODO: all other white spaces
    return char == char("\n") or char == char(" ");
}

function ascii_is_digit : (char : u8) -> bool = {
    return as_arithmetic(char) >= as_arithmetic(char("0")) and as_arithmetic(char) <= as_arithmetic(char("9"));
}

function ascii_is_alpha : (char : u8) -> bool = {
    let lower : i8 = as_arithmetic(ascii_to_lower(char));
    return lower >= as_arithmetic(char("a")) and lower <= as_arithmetic(char("z"));
}

function ascii_is_alphanumeric : (char : u8) -> bool = {
    return ascii_is_digit(char) or ascii_is_alpha(char);
}

function ascii_to_lower : (char : u8) -> u8 = {
    // TODO: support char literals
    if as_arithmetic(char) >= as_arithmetic(char("A")) and as_arithmetic(char) <= as_arithmetic(char("Z")) {
        return char | __builtin_narrow<u8>(0x20);
    }
    return char;
}

function ascii_to_upper : (char : u8) -> u8 = {
    // TODO: support char literals
    if as_arithmetic(char) >= as_arithmetic(char("a")) or as_arithmetic(char) <= as_arithmetic(char("z")) {
        return char ^ __builtin_narrow<u8>(0x20);
    }
    return char;
}

function ascii_is_matched_pair : (lhs : u8, rhs : u8) -> bool = {
    if lhs == char("(") and rhs == char(")") {
        return true;
    }
    if lhs == char("[") and rhs == char("]") {
        return true;
    }
    if lhs == char("<") and rhs == char(">") {
        return true;
    }
    if lhs == char("{") and rhs == char("}") {
        return true;
    }
    return false;
}

function char : (literal : u8[1]&) -> u8 = {
    // TODO: add char literals
    return literal[0];
}

function[Int] unstringify : (view : StringView) -> TypeIf<Int, IsIntegral<Int>> = {
    mut result : Int = 0;

    for char in view:get_iter() {
        runtime_assert(ascii_is_digit(char));
        result *= 10;
        result += as_arithmetic(char) - as_arithmetic(char("0"));
    }

    return result;
}

@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "iterators.c3"
@require_once "util.c3"
@require_once "wrappers.c3"

typedef StringView : Span<u8>

function [@Len] sv : (string : u8[@Len]&) -> StringView = {
    return make_span(&string);
}

@operator == : (lhs : StringView, rhs : StringView) -> bool = {
    if lhs.length != rhs.length {
        return false;
    }

    if ref_to_addr(&lhs.data) == ref_to_addr(&rhs.data) {
        // Short circuit when the memory overlaps
        return true;
    }

    for i in range(lhs.length) {
        if lhs.data[i] != rhs.data[i] {
            return false;
        }
    }
    return true;
}

typedef CString : { data : u8[&] }

function length : (str: CString&) -> isize = {
    let len : isize = 0;
    const null : u8[1]& = "\0"; // TODO: add char

    while str.data[len] != null[0] {
        len = len + 1;
    }

    return len;
}

function [@Len] make_cstring : (literal : u8[@Len]&) -> CString = {
    runtime_assert(literal[@Len - 1] == 0x00);
    return {&literal};
}

function ascii_is_white_space : (char : u8) -> bool = {
    // TODO: ||
    // TODO: all other white spaces
    return (char == "\n"[0]) | (char == " "[0]);
}

function ascii_is_digit : (char : u8) -> bool = {
    // TODO: &&
    return (as_arithmetic(char) >= as_arithmetic("0"[0])) & (as_arithmetic(char) <= as_arithmetic("9"[0]));
}

function ascii_is_alpha : (char : u8) -> bool = {
    // TODO: &&
    let lower : i8 = as_arithmetic(ascii_to_lower(char));
    return (lower >= as_arithmetic("a"[0])) & (lower <= as_arithmetic("z"[0]));
}

function ascii_is_alphanumeric : (char : u8) -> bool = {
    return ascii_is_digit(char) | ascii_is_alpha(char);
}

function ascii_to_lower : (char : u8) -> u8 = {
    // TODO: support &&
    // TODO: support char literals
    if (as_arithmetic(char) >= as_arithmetic("A"[0])) & (as_arithmetic(char) <= as_arithmetic("Z"[0])) {
        return char | __builtin_narrow<u8>(0x20);
    }
    return char;
}

function ascii_to_upper : (char : u8) -> u8 = {
    // TODO: support &&
    // TODO: support char literals
    if (as_arithmetic(char) >= as_arithmetic("a"[0])) | (as_arithmetic(char) <= as_arithmetic("z"[0])) {
        return char ^ __builtin_narrow<u8>(0x20);
    }
    return char;
}

@require_once "assert.c3"
@require_once "arithmetic.c3"

typedef CString : { data : u8[&] }

function length : (str: CString&) -> isize = {
    let len : isize = 0;
    const null : u8[1]& = "\0"; // TODO: add char

    while str.data[len] != null[0] {
        len = len + 1;
    }

    return len;
}

// TODO: this is kinda ugly, we need partial functions
function [@Len] make_cstring : (literal : u8[@Len]&) -> CString = {
    runtime_assert(literal[@Len - 1] == 0x00);
    return {&literal};
}

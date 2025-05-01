@require_once "std/assert.c3"
@require_once "std/span.c3"

function [@Len] v : (string : u8[@Len]&) -> View<u8> = {
    return make<View<u8>>(&string);
}

function main : () -> int = {
    // Test equality.
    runtime_assert(v("") == v(""));
    runtime_assert(v("a") != v(""));
    runtime_assert(v("a") == v("a"));
    runtime_assert(v("a") != v("aa"));
    runtime_assert(v("aaaa") != v("aaab"));
    runtime_assert(v("aaaa") == v("aaaa"));

    // Test inequality.
    runtime_assert(v("") < v("a"));
    runtime_assert(v("a") < v("b"));
    runtime_assert(v("abc") < v("abcd"));
    runtime_assert(v("abc") < v("bcd"));
    runtime_assert(v("abc") < v("bc"));
    runtime_assert(v("a") < v("aa"));

    return 0;
}

/// @COMPILE
/// @RUN

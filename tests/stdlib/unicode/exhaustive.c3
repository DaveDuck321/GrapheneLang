@require_once "std/unicode.c3"
@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/iterators.c3"
@require_once "std/span.c3"
@require_once "std/assert.c3"
@require_once "std/memory.c3"
@require_once "std/vector.c3"

function try_encode_and_decode : (original_scalar: UnicodeScalar) -> void = {
    mut buffer: u8[4];
    let encoded: View<u8> = encode_to_utf8(original_scalar, &mut buffer);

    mut decoder: UTF8DecodeIter = make<UTF8DecodeIter>(encoded);
    let decoded: UnicodeScalar = decoder:__builtin_get_next();
    runtime_assert(!decoder:has_next());
    runtime_assert(decoded == original_scalar);
}

// Checks that every Unicode scalar will correctly encode and decode
// This makes no attempt to verify incorrect utf-8 encodings are correctly handled
function main : () -> int = {
    // U+0000..U+D7FF
    for char in range(as_arithmetic(0x0000), as_arithmetic(0xD7FF) + 1)
    {
        try_encode_and_decode({as_logical(char)});
    }

    // U+D800–U+DFFF is reserved for the surrogate area

    // U+E000..U+FFFF
    for char in range(as_arithmetic(0xE000), as_arithmetic(0xFFFF) + 1)
    {
        try_encode_and_decode({as_logical(char)});
    }

    // U+10000..U+10FFFF
    for char in range(as_arithmetic(0x10000), as_arithmetic(0x10FFFF) + 1)
    {
        try_encode_and_decode({as_logical(char)});
    }

    return 0;
}


/// @COMPILE
/// @RUN

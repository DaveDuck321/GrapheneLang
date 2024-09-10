@require_once "std/unicode.c3"
@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/span.c3"
@require_once "std/assert.c3"
@require_once "std/memory.c3"
@require_once "std/vector.c3"

function read_file : ( path: CString ) -> View<u8> = {
    let file : Optional<File> = open(path, O_RDONLY());
    runtime_assert(file:has_value());

    let map : Optional<MappedFile> = file:data():read();
    file:data():close();
    runtime_assert(map:has_value());

    return map:data().buffer;
}

function main : () -> int = {
    let utf8_file       : View<u8> = read_file(make<CString>("stdlib/unicode/utf8.txt\0"));
    let utf16_be_file   : View<u8> = read_file(make<CString>("stdlib/unicode/utf16-be.txt\0"));
    let utf16_le_file   : View<u8> = read_file(make<CString>("stdlib/unicode/utf16-le.txt\0"));

    mut chars_processed : int = 0;

    mut utf16_be_iter : UTF16DecodeIter = make<UTF16DecodeIter>(utf16_be_file);
    mut utf16_le_iter : UTF16DecodeIter = make<UTF16DecodeIter>(utf16_le_file);

    // Correct detection of endianness?
    runtime_assert(!utf16_be_iter.is_little_endian);
    runtime_assert(utf16_le_iter.is_little_endian);

    mut allocator: Allocator = initialize_allocator();
    mut reconstructed_utf8_file : Vector<u8> = make<Vector<u8>>(&mut allocator);

    // All parsers decode the same scalars?
    for utf8_scalar in make<UTF8DecodeIter>(utf8_file) {
        runtime_assert(utf16_be_iter:has_next());
        runtime_assert(utf16_le_iter:has_next());

        let utf16_be_scalar : UnicodeScalar = utf16_be_iter:__builtin_get_next();
        let utf16_le_scalar : UnicodeScalar = utf16_le_iter:__builtin_get_next();

        runtime_assert(utf8_scalar == utf16_be_scalar);
        runtime_assert(utf8_scalar == utf16_le_scalar);

        {
            mut buffer: u8[4];
            let result: View<u8> = encode_to_utf8(utf16_be_scalar, &mut buffer);
            reconstructed_utf8_file:extend_back(result:get_iter());
        }

        chars_processed += 1;
    }

    // All parsers terminate on the same character?
    runtime_assert(!utf16_be_iter:has_next());
    runtime_assert(!utf16_le_iter:has_next());

    // As calculated by python's text-mode len(file.read())
    runtime_assert(chars_processed == 180);

    // https://www.compart.com/en/unicode/U+1F60E
    runtime_assert(make<UnicodeScalar>(sv("😎")).value == 0x0001F60E);

    // Was encoding successful?
    runtime_assert(utf8_file == reconstructed_utf8_file:view());
    return 0;
}


/// @COMPILE
/// @RUN

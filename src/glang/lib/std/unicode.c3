@require_once "std/string.c3"
@require_once "std/iterators.c3"
@require_once "std/assert.c3"
@require_once "std/span.c3"
@require_once "std/wrappers.c3"
@require_once "std/arithmetic.c3"

typedef UnicodeScalar : {
    value: u32
}

typedef UTF8DecodeIter : {
    underlying: ContiguousContainerIter<View<u8>, u8&>,
}

typedef UTF16DecodeIter : {
    underlying: ContiguousContainerIter<View<u8>, u8&>,
    is_little_endian: bool,
}

function invalid_unicode_scalar : () -> UnicodeScalar = {
    // U+FFFD: 'REPLACEMENT CHARACTER'
    return { 0xFFFD };
}

function make<UnicodeScalar> : (char: StringView) -> UnicodeScalar = {
    // Graphene source is utf-8
    mut iter : UTF8DecodeIter = { char.buffer:get_iter() };
    let scalar: UnicodeScalar = iter:__builtin_get_next();
    runtime_assert(!iter:has_next());
    return scalar;
}

// TODO parsing fix from: https://github.com/DaveDuck321/GrapheneLang/pull/164
// typedef HasSpaceship<UnicodeScalar> : TrueType
//
// @operator <=> : (self : UnicodeScalar, other : UnicodeScalar) -> bool = {
//     return as_arithmetic(self.value) - as_arithmetic(other.value);
// }

@operator == : (self : UnicodeScalar, other : UnicodeScalar) -> bool = {
    return self.value == other.value;
}

@operator != : (self : UnicodeScalar, other : UnicodeScalar) -> bool = {
    return self.value != other.value;
}

function make<UTF8DecodeIter> : ( string: StringView ) -> UTF8DecodeIter = {
    return { string.buffer:get_iter() };
}

function make<UTF8DecodeIter> : ( buffer: View<u8> ) -> UTF8DecodeIter = {
    return { buffer:get_iter() };
}

function make<UTF16DecodeIter> : ( buffer: View<u8> ) -> UTF16DecodeIter = {
    // Unicode-15.0.0
    // In the UTF-16 encoding scheme, an initial byte sequence corresponding to
    // U+FEFF is interpreted as a byte order mark; it is used to distinguish between
    // the two byte orders. An initial byte sequence <FE FF> indicates big-endian
    // order, and an initial byte sequence <FF FE> indicates little-endian order. The
    // BOM is not considered part of the content of the text
    mut iter : ContiguousContainerIter<View<u8>, u8&> = buffer:get_iter();
    mut is_little_endian : bool = false;
    if buffer:length() >= 2 {
        // The UTF-16 encoding scheme may or may not begin with a BOM. However,
        // when there is no BOM, and in the absence of a higher-level protocol, the byte
        // order of the UTF-16 encoding scheme is big-endian.
        if buffer:get(0) == 0xfe and buffer:get(1) == 0xff {
            is_little_endian = false;
            iter:__builtin_get_next();
            iter:__builtin_get_next();
        }

        if buffer:get(0) == 0xff and buffer:get(1) == 0xfe {
            is_little_endian = true;
            iter:__builtin_get_next();
            iter:__builtin_get_next();
        }
    }
    return { iter, is_little_endian };
}

// --------- UTF8 -----------
// https://www.unicode.org/versions/Unicode4.0.0/ch03.pdf#G31703
function enum_utf8_invalid : () -> u32 = { return 0x00; }
function enum_utf8_encoding_ascii : () -> u32 = { return 0x01; }
function enum_utf8_encoding_2_byte : () -> u32 = { return 0x02; }
function enum_utf8_encoding_3_byte_E0 : () -> u32 = { return 0x03; }
function enum_utf8_encoding_3_byte_E1_EC : () -> u32 = { return 0x04; }
function enum_utf8_encoding_3_byte_ED : () -> u32 = { return 0x05; }
function enum_utf8_encoding_3_byte_EE_EF : () -> u32 = { return 0x06; }
function enum_utf8_encoding_4_byte_F0 : () -> u32 = { return 0x07; }
function enum_utf8_encoding_4_byte_F1_F3 : () -> u32 = { return 0x08; }
function enum_utf8_encoding_4_byte_F4 : () -> u32 = { return 0x09; }

function get_utf8_encoding_length : (utf8_encoding_type: u32) -> isize = {
    if utf8_encoding_type == enum_utf8_invalid() or utf8_encoding_type == enum_utf8_encoding_ascii() {
        return 1;
    }

    if utf8_encoding_type == enum_utf8_encoding_2_byte() {
        return 2;
    }

    if utf8_encoding_type == enum_utf8_encoding_3_byte_E0() or
        utf8_encoding_type == enum_utf8_encoding_3_byte_E1_EC() or
        utf8_encoding_type == enum_utf8_encoding_3_byte_ED() or
        utf8_encoding_type == enum_utf8_encoding_3_byte_EE_EF()
    {
        return 3;
    }

    if utf8_encoding_type == enum_utf8_encoding_4_byte_F0() or
        utf8_encoding_type == enum_utf8_encoding_4_byte_F1_F3() or
        utf8_encoding_type == enum_utf8_encoding_4_byte_F4()
    {
        return 4;
    }

    runtime_assert(false);
    return 0;
}

function is_in_range : (value: u32, lower_inclusive: u32, upper_inclusive: u32) -> bool = {
    // Inclusive range better matches the spec comments
    // Use u32 here since we don't have u8 literals
    return value == lower_inclusive or value == upper_inclusive or
            (__builtin_is_less_than(value, upper_inclusive) and __builtin_is_greater_than(value, lower_inclusive));
}

function get_utf8_encoding_type : (first_byte: u8) -> u32 = {
    // Unicode 15.0.0 encoding table:
    // U+0000..U+007F       00..7F
    // U+0080..U+07FF       C2..DF  80..BF
    // U+0800..U+0FFF       E0      A0..BF  80..BF
    // U+1000..U+CFFF       E1..EC  80..BF  80..BF
    // U+D000..U+D7FF       ED      80..9F  80..BF
    // U+E000..U+FFFF       EE..EF  80..BF  80..BF
    // U+10000..U+3FFFF     F0      90..BF  80..BF  80..BF
    // U+40000..U+FFFFF     F1..F3  80..BF  80..BF  80..BF
    // U+100000..U+10FFFF   F4      80..8F  80..BF  80..BF

    // U+0000..U+007F:      00..7F
    if is_in_range(first_byte, 0x00, 0x7f) {
        return enum_utf8_encoding_ascii();
    }

    // U+0080..U+07FF:      C2..DF
    if is_in_range(first_byte, 0xc2, 0xdf) {
        return enum_utf8_encoding_2_byte();
    }

    // U+0800..U+0FFF:      E0
    if first_byte == 0xe0 {
        return enum_utf8_encoding_3_byte_E0();
    }

    // U+1000..U+CFFF:      E1..EC
    if is_in_range(first_byte, 0xe1, 0xec) {
        return enum_utf8_encoding_3_byte_EE_EF();
    }

    // U+D000..U+D7FF:      ED
    if first_byte == 0xed {
        return enum_utf8_encoding_3_byte_ED();
    }

    // U+E000..U+FFFF:      EE..EF
    if is_in_range(first_byte, 0xee, 0xef) {
        return enum_utf8_encoding_3_byte_E1_EC();
    }

    // U+10000..U+3FFFF:    F0
    if first_byte == 0xf0 {
        return enum_utf8_encoding_4_byte_F0();
    }

    // U+40000..U+FFFFF:    F1..F3
    if is_in_range(first_byte, 0xf1, 0xf3) {
        return enum_utf8_encoding_4_byte_F1_F3();
    }

    // U+100000..U+10FFFF:  F4
    if first_byte == 0xf4 {
        return enum_utf8_encoding_4_byte_F4();
    }

    return enum_utf8_invalid();
}

function is_trailing_utf8_byte_in_valid_range : (byte: u8) -> bool = {
    // Trailing bytes (which are not the second byte) always fall within 80..BF
    return is_in_range(byte, 0x80, 0xbf);
}

function is_2nd_utf8_byte_in_valid_range : (enum_utf8_encoding: u32, byte: u8) -> bool = {
    // U+0800..U+0FFF:      E0  A0..BF
    if enum_utf8_encoding == enum_utf8_encoding_3_byte_E0() {
        return is_in_range(byte, 0xa0, 0xbf);
    }

    // U+D000..U+D7FF:      ED  80..9F  80..BF
    if enum_utf8_encoding == enum_utf8_encoding_3_byte_ED() {
        return is_in_range(byte, 0x80, 0x9f);
    }

    // U+10000..U+3FFFF:    F0  90..BF  80..BF  80..BF
    if enum_utf8_encoding == enum_utf8_encoding_4_byte_F0() {
        return is_in_range(byte, 0x90, 0xbf);
    }

    // U+100000..U+10FFFF:  F4  80..8F  80..BF  80..BF
    if enum_utf8_encoding == enum_utf8_encoding_4_byte_F4() {
        return is_in_range(byte, 0x80, 0x8f);
    }

    // Otherwise this isn't a special byte
    return is_trailing_utf8_byte_in_valid_range(byte);
}

function unsafe_decode_utf8_bytes : (bytes: u8[1]) -> UnicodeScalar = {
    return { bytes[0] };
}

function unsafe_decode_utf8_bytes : (bytes: u8[2]) -> UnicodeScalar = {
    return { ((bytes[0] & 0x1f) << 6) | (bytes[1] & 0x3f) };
}

function unsafe_decode_utf8_bytes : (bytes: u8[3]) -> UnicodeScalar = {
    return { ((bytes[0] & 0x0f) << 12) | ((bytes[1] & 0x3f) << 6) | (bytes[2] & 0x3f) };
}

function unsafe_decode_utf8_bytes : (bytes: u8[4]) -> UnicodeScalar = {
    return {
        ((bytes[0] & 0x07) << 18) |
        ((bytes[1] & 0x3f) << 12) |
        ((bytes[2] & 0x3f) << 6)  |
        (bytes[3] & 0x3f)
    };
}

@implicit get_next : (self: UTF8DecodeIter mut&) -> UnicodeScalar  = {
    let first_byte : u8 = self.underlying:__builtin_get_next();

    let encoding_type : u32 = get_utf8_encoding_type(first_byte);
    let encoding_length: isize = get_utf8_encoding_length(encoding_type);

    // Unicode-15.0.0
    // If the converter encounters an ill-formed UTF-8 code unit sequence
    // which starts with a valid first byte, but which does not continue with valid successor bytes
    // (see Table 3-7), it must not consume the successor bytes as part of the ill-formed subse-
    // quence whenever those successor bytes themselves constitute part of a well-formed UTF-8
    // code unit subsequence.

    // We apply the optional: U+FFFD Substitution of Maximal Subparts algorithm
    // Whenever an unconvertible offset is reached during conversion of a code unit sequence:
    // 1. The maximal subpart at that offset is replaced by a single U+FFFD
    // 2. The conversion proceeds at the offset immediately after the maximal subpart.

    // Byte 1
    if encoding_type == enum_utf8_invalid() {
        return invalid_unicode_scalar();
    }

    if encoding_length == 1 {
        return unsafe_decode_utf8_bytes({first_byte});
    }

    // Byte 2
    if !self.underlying:has_next() {
        return invalid_unicode_scalar(); // TODO: iterators should be nicer than this
    }

    let second_byte : u8 = self.underlying:peak();
    if !is_2nd_utf8_byte_in_valid_range(encoding_type, second_byte) {
        return invalid_unicode_scalar();
    }

    self.underlying:__builtin_get_next();
    if encoding_length == 2 {
        return unsafe_decode_utf8_bytes({first_byte, second_byte});
    }

    // Byte 3
    if !self.underlying:has_next() {
        return invalid_unicode_scalar();
    }

    let third_byte : u8 = self.underlying:peak();
    if !is_trailing_utf8_byte_in_valid_range(third_byte) {
        return invalid_unicode_scalar();
    }

    self.underlying:__builtin_get_next();
    if encoding_length == 3 {
        return unsafe_decode_utf8_bytes({first_byte, second_byte, third_byte});
    }

    // Byte 4
    if !self.underlying:has_next() {
        return invalid_unicode_scalar();
    }

    let forth_byte : u8 = self.underlying:peak();
    if !is_trailing_utf8_byte_in_valid_range(forth_byte) {
        return invalid_unicode_scalar();
    }

    self.underlying:__builtin_get_next();
    return unsafe_decode_utf8_bytes({first_byte, second_byte, third_byte, forth_byte});
}

@implicit has_next : (self : UTF8DecodeIter mut&) -> bool = {
    return self.underlying:has_next();
}

// --------- UTF16 -----------
function get_next_codepoint : (self: UTF16DecodeIter mut&) -> Optional<u16> = {
    if !self.underlying:has_next() { return make<Optional<u16>>(); }
    let first_byte : u8 = self.underlying:__builtin_get_next();

    if !self.underlying:has_next() { return make<Optional<u16>>(); }
    let second_byte : u8 = self.underlying:__builtin_get_next();

    if self.is_little_endian {
        return make<Optional<u16>>(first_byte | (as_u16(second_byte) << Narrow<i16>(8)));
    } else {
        return make<Optional<u16>>(second_byte | (as_u16(first_byte) << Narrow<i16>(8)));
    }
}

@implicit get_next : (self: UTF16DecodeIter mut&) -> UnicodeScalar = {
    let maybe_codepoint1 : Optional<u16> = self:get_next_codepoint();
    if !maybe_codepoint1:has_value() {
        return invalid_unicode_scalar();
    }

    let codepoint1 : u16 = maybe_codepoint1:data();

    // Is this the second part of a surrogate pair?
    if codepoint1 & 0xfc00 == 0xdc00 {
        return invalid_unicode_scalar();
    }

    // Is this a standalone codepoint?
    if codepoint1 & 0xfc00 != 0xd800 {
        return { codepoint1 };
    }

    // None of above, it must be the first part of a surrogate pair
    let maybe_codepoint2 : Optional<u16> = self:get_next_codepoint();
    if !maybe_codepoint2:has_value() {
        return invalid_unicode_scalar();
    }

    let codepoint2 : u16 = maybe_codepoint2:data();

    let offset_scalar : u32 = (as_u32(codepoint1 & 0x3ff) << 10) | as_u32(codepoint2 & 0x3ff);
    return { as_logical(as_arithmetic(offset_scalar) + as_arithmetic(0x10000)) };
}

@implicit has_next : (self : UTF16DecodeIter mut&) -> bool = {
    return self.underlying:has_next();
}

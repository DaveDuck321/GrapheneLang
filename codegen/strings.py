from .user_facing_errors import InvalidEscapeSequence

# Based on https://en.cppreference.com/w/cpp/language/escape.
# We omit \' because we don't have single-quoted strings, and \? because
# we don't have trigraphs.
_ESCAPE_SEQUENCES_TABLE = {
    '"': chr(0x22),
    "\\": chr(0x5C),
    "0": chr(0x00),
    "a": chr(0x07),
    "b": chr(0x08),
    "f": chr(0x0C),
    "n": chr(0x0A),
    "r": chr(0x0D),
    "t": chr(0x09),
    "v": chr(0x0B),
}


def encode_string(string: str) -> tuple[str, int]:
    # LLVM is a bit vague on what is acceptable, but we definitely need to
    # escape non-printable characters and double quotes with "\xx", where
    # xx is the hexadecimal representation of each byte. We also parse
    # escape sequences here.
    # XXX we're using utf-8 for everything.

    first: int = 0
    byte_len: int = 0
    buffer: list[str] = []

    def encode_char(char: str) -> None:
        nonlocal byte_len

        utf8_bytes = char.encode("utf-8")

        for byte in utf8_bytes:
            buffer.append(f"\\{byte:0>2x}")

        byte_len += len(utf8_bytes)

    def consume_substr(last: int) -> None:
        nonlocal byte_len

        # Append the substr as-is instead of the utf-8 representation, as
        # python will encode it anyway when we write to the output stream.
        substr = string[first:last]
        buffer.append(substr)
        # FIXME there must be a better way.
        byte_len += len(substr.encode("utf-8"))

    chars = iter(enumerate(string))
    for idx, char in chars:
        if char == "\\":
            # Consume up to the previous character.
            consume_substr(idx)

            # Should never raise StopIteration, as the grammar guarantees
            # that we don't end a string with a \.
            _, escaped_char = next(chars)

            if escaped_char not in _ESCAPE_SEQUENCES_TABLE:
                raise InvalidEscapeSequence(escaped_char)

            # Easier if we always encode the representation of the escape
            # sequence.
            encode_char(_ESCAPE_SEQUENCES_TABLE[escaped_char])

            # Start from the character after the next one.
            first = idx + 2
        elif not char.isprintable():
            # Consume up to the previous character.
            consume_substr(idx)

            # Escape current character.
            encode_char(char)

            # Start from the next character.
            first = idx + 1

    # Consume any remaining characters.
    consume_substr(len(string))

    return "".join(buffer), byte_len

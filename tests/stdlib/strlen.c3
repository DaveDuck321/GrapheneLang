@require_once "std/string.c3"
@require_once "std/assert.c3"

function main : () -> int = {
    runtime_assert(strlen("\0") == 0);
    runtime_assert(strlen("abc\0") == 3);
    runtime_assert(strlen("\n\n\n\n\n\n\n\n\n\n\0") == 10);

    return 0;
}

/// @COMPILE
/// @RUN

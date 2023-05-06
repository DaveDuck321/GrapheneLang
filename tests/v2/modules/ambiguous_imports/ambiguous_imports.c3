@require_once "_import.c3";

function main : () -> int = {
    return test();
};

/// @COMPILE -Iincluded; EXPECT ERR
/// File '*.c3', line 1, in '@require_once "_import.c3"'
/// Error: file '_import.c3' is ambiguous, possible candidates are:
/// - */_import.c3
/// - */included/_import.c3

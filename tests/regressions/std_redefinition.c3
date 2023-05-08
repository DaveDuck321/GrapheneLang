@require_once "std/type_traits.c3";

typedef EnableIf<TrueType>: {};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'typedef'
/// Error: multiple definitions of type 'EnableIf<TrueType>'

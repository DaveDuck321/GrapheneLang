@require_once "std/type_traits.c3"

// Comment so that the EnableIfs aren't on the same line
typedef EnableIf<TrueType>: {}

/// @COMPILE; EXPECT ERR
/// File '*/std_redefinition.c3', line 4, in 'typedef'
/// Error: multiple definitions of type 'EnableIf<TrueType>':
/// - typedef EnableIf<TrueType> : {}      (File '*.c3', line ?)
/// - typedef EnableIf<TrueType> : {}      (File '*.c3', line ?)

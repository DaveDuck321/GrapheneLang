typedef int : i8

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'typedef'
/// Error: multiple definitions of type 'int':
/// - int                   (builtin)
/// - typedef int : i8      (File '*.c3', line 1)

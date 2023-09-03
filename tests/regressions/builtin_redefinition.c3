typedef int : i8

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'typedef'
/// Error: type 'int' is ambiguous. Equally good candidates are:
/// - int      (builtin)
/// - int      (File '*.c3', line 1)

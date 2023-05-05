foreign foo : (x: void) -> void;

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'foreign signature'
/// Error: argument 'x' cannot have type 'void'

function[T, Y] foo : (x : T<Y>) -> void = {};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'function[...] signature'
/// Error: generic 'T' has a generic annotation (generic types may not have additional generic annotations)

function main : () -> void = {
    const a : int;
}
/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2
/// const a : int;
/// ^
/// Error: invalid syntax, a constant cannot be declared uninitialized

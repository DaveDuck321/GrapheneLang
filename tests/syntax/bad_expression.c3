function main : () -> int = {
    #;
}
/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2
/// function main : () -> int = {
/// #;
/// ^
/// Error: invalid syntax, unexpected symbol '#', cannot start an expression

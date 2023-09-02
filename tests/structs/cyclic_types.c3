typedef Node : {
    self : Node,
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1, in 'typedef'
/// Error: cannot solve type hierarchy; type 'Node' has a cyclic definition

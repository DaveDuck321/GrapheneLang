typedef[T] MyType : bool
typedef MyType<int> : int
typedef MyType<u8> : bool
typedef MyType<u8> : int

/// @COMPILE; EXPECT ERR
/// File *.c3', line 4, in 'typedef'
/// Error: multiple definitions of type 'MyType<u8>'

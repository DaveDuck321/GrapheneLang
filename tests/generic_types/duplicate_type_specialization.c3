typedef[T] MyType : bool
typedef MyType<int> : int
typedef MyType<u8> : bool
typedef MyType<u8> : int

/// @COMPILE; EXPECT ERR
/// File *.c3', line 3, in 'typedef'
/// Error: multiple definitions of type 'MyType<u8>':
/// - typedef MyType<u8> : bool      (File '*.c3', line 3)
/// - typedef MyType<u8> : int       (File '*.c3', line 4)

typedef[T] MyType : bool
typedef MyType<int> : int
typedef MyType<u8> : bool
typedef MyType<u8> : int

/// @COMPILE; EXPECT ERR
/// File *.c3', line 3, in 'typedef'
/// Error: type 'MyType<u8>' is ambiguous. Equally good candidates are:
/// - typedef MyType<u8> : bool      (File '*.c3', line 3)
/// - typedef MyType<u8> : bool      (File '*.c3', line 4)

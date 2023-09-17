// Error: generic 'T' has a generic annotation (generic types may not have
// additional generic annotations)
typedef[T, U] struct_t: {a: T<U>}

function main: () -> int = {
    mut s: struct_t<int, int>; // unused typedef's are not parsed

    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3
/// Error: generic 'T' has a generic annotation (generic types may not have additional generic annotations)

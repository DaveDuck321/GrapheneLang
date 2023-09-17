typedef MyStruct : {a: int, x: int}

function[T] get_x : (struct : T) -> int = {
    return struct.x;
}

function main : () -> int = {
    return get_x({11, 12});
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 8, in 'function main: () -> int'
/// Error: cannot deduce generic type 'T' in function 'get_x<T> : (T) -> int', manual specialization is required

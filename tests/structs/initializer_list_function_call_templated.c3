function[T] get_x : (struct : T) -> int = {
    return struct.x;
}

function main : () -> int = {
    return get_x({.x = 11, .y = 12});
}

/// @COMPILE
/// @RUN; EXPECT 11

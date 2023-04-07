function a : (x: i32, y: i64) -> int = { return 0; };
function a : (x: i64, y: i32) -> int = { return 1; };

function main: () -> int = {
    return a(0, 0);
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: overload resolution for function call 'a(int, int)' is ambiguous.
/// Equally good candidates are
/// - function a: (i32, i64) -> int
/// - function a: (i64, i32) -> int

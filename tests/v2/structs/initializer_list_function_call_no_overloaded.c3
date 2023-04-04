typedef MyStruct1 : {a: int, x: int, b: int};
typedef MyStruct2 : {a: int, x: int};

function get_x : (struct : MyStruct1) -> int = {
    return struct.x;
};

function get_x : (struct : MyStruct2) -> int = {
    return struct.x;
};

function main : () -> int = {
    return get_x({11});
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 13, in 'function main: () -> int'
/// Error: overload resolution for function call 'get_x({int})' failed
/// Available overloads:
/// - function get_x: (MyStruct1) -> int
/// - function get_x: (MyStruct2) -> int

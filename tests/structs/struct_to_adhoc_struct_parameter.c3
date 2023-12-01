typedef Struct : {a : int, b : int}

function test : (param : {a : int, b : int}) -> int = {
    return param.a;
}

function main : () -> int = {
    let struct : Struct = {1, 1};
    return test(struct);
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 9, in 'function main: () -> int'
/// Error: overload resolution for function call 'test(Struct)' failed
/// Available overloads:
/// - function test : ({ a: int, b: int }) -> int

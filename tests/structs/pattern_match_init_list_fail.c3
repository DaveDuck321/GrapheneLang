function [U, V] test : (param : {a : U, b : V}) -> U = {
    return param.a;
}

function main : () -> int = {
    // Fails because layout would be incorrect
    return test({.b = 50, .a = 100});
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 7, in 'function main: () -> int'
/// Error: overload resolution for function call 'test({b: int, a: int})' failed
/// Available overloads:
/// - function [[]U, V[]] test<U, V> : ({ a: U, b: V }) -> U

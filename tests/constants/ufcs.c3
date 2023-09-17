// I don't particually like this, I think I'll change it to something smarter later

typedef A : {
    a : int
}


function run : (a : A&) -> int = {
    return a.a;
}

function main : () -> int = {
    mut a : A = {1};
    return a:run();
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 14, in 'function main: () -> int'
/// Error: overload resolution for function call 'run(A mut&)' failed
/// Available overloads:
/// - function run : (A&) -> int

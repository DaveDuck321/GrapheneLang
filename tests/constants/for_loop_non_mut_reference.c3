typedef Iter : {}

@implicit has_next : (iter : Iter&) -> bool = {return false;}
@implicit get_next : (iter : Iter&) -> int = {return 1;}

function main : () -> int = {
    let iter : Iter = {};
    for i in iter {

    }
    return 0;
}

///  @COMPILE; EXPECT ERR
/// File '*.c3', line 8, in 'function main: () -> int'
/// Error: overload resolution for function call '__builtin_has_next(Iter mut&)' failed
/// Available overloads:
/// - function __builtin_has_next : (Iter&) -> bool

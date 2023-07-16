typedef A : { x : int }
typedef B : A

function foo : () -> B = {
    let a : A = {1};
    return a;
}

function main : () -> int = {
    // This would erroneously cause a codegen error, even though our typesystem worked
    // `error: value doesn't match function result type '%type.__T_B = type { i32 }'`
    let a : A = foo();
    return 0;
}

/// @COMPILE
/// @RUN; EXPECT 0

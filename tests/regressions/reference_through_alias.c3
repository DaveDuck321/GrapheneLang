typedef [T] Identity : T
typedef MyType : {}

function takes_my_type : (type : Identity<MyType>&) -> void = {}

function main : () -> int = {
    let a : Identity<MyType> = {};
    takes_my_type(&a);
    return 0;
}

/// @COMPILE
/// @RUN

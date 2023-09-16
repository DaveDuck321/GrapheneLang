typedef MyStruct : {a: int, x: int}

function[T] forward_ref : (arg: T mut&) -> T mut& = {
    return &mut arg;
}

function main : () -> int = {
    mut struct : MyStruct = {42, 10};
    forward_ref<MyStruct>(&mut struct).x = 11;

    return forward_ref<MyStruct>(&mut struct).x;
}

/// @COMPILE
/// @RUN; EXPECT 11

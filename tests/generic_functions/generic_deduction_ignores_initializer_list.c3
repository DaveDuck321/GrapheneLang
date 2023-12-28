typedef A : {}
typedef B : {}

function make<A> : (val : A) -> A = { return {};}

function [@Len] make<B> : (buffer : u8[@Len]&) -> B = {
    return {};
}

function main : () -> int = {
    let a : A = make<A>({});
    return 0;
}

/// @COMPILE
/// @RUN

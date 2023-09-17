typedef Iter : {}

@implicit has_next : (iter : Iter mut&) -> bool = {return false;}
@implicit get_next : (iter : Iter mut&) -> int = {return 1;}

function main : () -> int = {
    let iter : Iter = {};
    for i in iter { // Iterator is copied here

    }
    return 0;
}

/// @COMPILE
/// @RUN

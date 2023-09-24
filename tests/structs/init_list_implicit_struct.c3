function main : () -> int = {
    let a : {val : int, sign : bool} = {0, false};
    return a.val;
}

/// @COMPILE
/// @RUN

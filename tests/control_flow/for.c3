@require_once "std/arithmetic.c3";

typedef Range : {lower: int, upper: int};

function range : (lower: int, upper: int) -> Range = {
    return {lower, upper};
};

@implicit has_next : (range: Range&) -> bool = {
    return range.upper > range.lower;
};

@implicit get_next : (range: Range&) -> int = {
    let return_value: int = range.lower;
    range.lower += 1;
    return return_value;
};

function main : () -> int = {
    let sum : int = 0;
    for (i in range(0, 10)) {
        sum += i;
    }
    return sum;
};


/// @COMPILE
/// @RUN; EXPECT 45

@require_once "arithmetic.c3"
@require_once "type_traits.c3"

typedef [T] Range : {lower : T, upper : T, step : T}

@implicit [T] get_next : (range: Range<T>&) -> T = {
    let next : T = range.lower;
    range.lower += range.step;
    return next;
}

@implicit [T] has_next : (range : Range<T>&) -> bool = {
    return range.lower < range.upper;
}

function [T] range : (upper : T) -> TypeIf<Range<T>, IsIntegral<T>> = {
    return {0, upper, 1};
}

function [T] range : (lower : T, upper : T) -> TypeIf<Range<T>, IsIntegral<T>> = {
    return {lower, upper, 1};
}

function [T] range : (lower : T, upper : T, step : T) -> TypeIf<Range<T>, IsIntegral<T>> = {
    return {lower, upper, step};
}

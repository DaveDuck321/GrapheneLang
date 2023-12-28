@require_once "std/type_traits.c3"

function [T] abs : (value : T) -> TypeIf<T, IsIntegral<T>> = {
    if value < 0 {
        return -value;
    } else {
        return value;
    }
}

function [T] abs : (value : T) -> TypeIf<T, IsFloating<T>> = {
    if value < 0.0 {
        return -value;
    } else {
        return value;
    }
}

function [T] max : (left : T, right : T) -> T = {
    if left < right {
        return right;
    } else {
        return left;
    }
}

function [T] min : (left : T, right : T) -> T = {
    if left < right {
        return left;
    } else {
        return right;
    }
}

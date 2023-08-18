@require_once "std/type_traits.c3"

function [T] abs : (value : T) -> TypeIf<T, IsIntegral<T>> = {
    if value < 0 {
        return -value;
    } else {
        return value;
    }
}

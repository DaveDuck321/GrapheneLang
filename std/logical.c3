@require_once "type_traits.c3";

@operator ! : (value : bool) -> bool = {
    if (value) {
        return false;
    }
    return true;
};

@operator [T] | : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_or(lhs, rhs);
};

@operator [T] & : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_and(lhs, rhs);
};

@operator [T] ^ : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_xor(lhs, rhs);
};

@operator [U, V] << : (value: U, shift_amount: V) -> TypeIf<U, Both<IsLogical<U>, IsIntegral<V>>> = {
    return __builtin_shift_left(value, shift_amount);
};

@operator [T] >> : (value: U, shift_amount: V) -> TypeIf<U, Both<IsLogical<U>, IsIntegral<V>>> = {
    return __builtin_shift_right(value, shift_amount);
};

@operator [T] == : (lhs: T, rhs: T) -> TypeIf<bool, IsLogical<T>> = {
    return __builtin_is_equal(lhs, rhs);
};

@operator [T] != : (lhs: T, rhs: T) -> TypeIf<bool, IsLogical<T>> = {
    return !(lhs == rhs);
};

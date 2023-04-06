@require_once "type_traits.c3";
@require_once "arithmetic.c3";

typedef[T] IsLogical : FalseType;
typedef IsLogical<bool> : TrueType;
typedef IsLogical<u8> : TrueType;
typedef IsLogical<u16> : TrueType;
typedef IsLogical<u32> : TrueType;
typedef IsLogical<u64> : TrueType;
typedef IsLogical<u128> : TrueType;

@operator [T] | : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_or(lhs, rhs);
};

@operator [T] & : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_and(lhs, rhs);
};

@operator [T] ^ : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_xor(lhs, rhs);
};

@operator [U, V] << : (value: U, shift_amount: V) -> TypeIf<U, Both<IsLogical<U>, IsArithmetic<V>>> = {
    return __builtin_shift_left(value, shift_amount);
};

@operator [T] >> : (value: U, shift_amount: V) -> TypeIf<U, Both<IsLogical<U>, IsArithmetic<V>>> = {
    return __builtin_shift_right(value, shift_amount);
};

@operator [T] == : (lhs: T, rhs: T) -> TypeIf<bool, IsLogical<T>> = {
    return __builtin_is_equal(lhs, rhs);
};

@operator [T] != : (lhs: T, rhs: T) -> TypeIf<bool, IsLogical<T>> = {
    if (lhs == rhs) {
        return false;
    }
    return true;
};

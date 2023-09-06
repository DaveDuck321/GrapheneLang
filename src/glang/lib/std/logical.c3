@require_once "assignments.c3"
@require_once "type_traits.c3"
@require_once "util.c3"

typedef HasTrivialAssignments<bool> : TrueType
typedef HasTrivialAssignments<u8> : TrueType
typedef HasTrivialAssignments<u16> : TrueType
typedef HasTrivialAssignments<u32> : TrueType
typedef HasTrivialAssignments<u64> : TrueType
typedef HasTrivialAssignments<u128> : TrueType

@operator ! : (value : bool) -> bool = {
    if value {
        return false;
    }
    return true;
}

@operator [T] ~ : (value: T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_not(value);
}

@operator [T] | : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_or(lhs, rhs);
}

@operator [T] & : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_and(lhs, rhs);
}

@operator [T] ^ : (lhs : T, rhs : T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_xor(lhs, rhs);
}

@operator [L, A] << : (value: L, shift_amount: A) -> TypeIf<L, Both<IsLogical<L>, IsIntegral<A>>> = {
    let ushift : L = as_logical(shift_amount);
    return __builtin_shift_left(value, ushift);
}

@operator [L, A] >> : (value: L, shift_amount: A) -> TypeIf<L, Both<IsLogical<L>, IsIntegral<A>>> = {
    let ushift : L = as_logical(shift_amount);
    return __builtin_shift_right(value, ushift);
}

@operator [T] == : (lhs: T, rhs: T) -> TypeIf<bool, IsLogical<T>> = {
    return __builtin_is_equal(lhs, rhs);
}

@operator [T] != : (lhs: T, rhs: T) -> TypeIf<bool, IsLogical<T>> = {
    return !(lhs == rhs);
}

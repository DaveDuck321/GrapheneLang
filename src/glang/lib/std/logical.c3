@require_once "assignments.c3"
@require_once "type_traits.c3"
@require_once "util.c3"

typedef [U, V] CommonLogicalType: TypeIf<U, Both<IsLogical<U>, IsLogical<V>>>
typedef CommonLogicalType<u8, u16> : u16
typedef CommonLogicalType<u8, u32> : u32
typedef CommonLogicalType<u8, u64> : u64
typedef CommonLogicalType<u8, u128> : u128
typedef CommonLogicalType<u16, u32> : u32
typedef CommonLogicalType<u16, u64> : u64
typedef CommonLogicalType<u16, u128> : u128
typedef CommonLogicalType<u32, u64> : u64
typedef CommonLogicalType<u32, u128> : u128
typedef CommonLogicalType<u64, u128> : u128

typedef [U, V] LogicalCompareResult : TypeIf<bool, Both<IsLogical<U>, IsLogical<V>>>
typedef [T] RegisterHasTrivialAssignments<Override<T>> : TypeIf<TrueType, IsLogical<T>>

@operator ! : (value : bool) -> bool = {
    if value {
        return false;
    }
    return true;
}

@operator [T] ~ : (value: T) -> TypeIf<T, IsLogical<T>> = {
    return __builtin_bitwise_not(value);
}

@operator [U, V] | : (lhs : U, rhs : V) -> CommonLogicalType<U, V> = {
    let lhs_widen: CommonLogicalType<U, V> = lhs;
    let rhs_widen: CommonLogicalType<U, V> = rhs;
    return __builtin_bitwise_or(lhs_widen, rhs_widen);
}

@operator [U, V] & : (lhs : U, rhs : V) -> CommonLogicalType<U, V> = {
    let lhs_widen: CommonLogicalType<U, V> = lhs;
    let rhs_widen: CommonLogicalType<U, V> = rhs;
    return __builtin_bitwise_and(lhs_widen, rhs_widen);
}

@operator [U, V] ^ : (lhs : U, rhs : V) -> CommonLogicalType<U, V> = {
    let lhs_widen: CommonLogicalType<U, V> = lhs;
    let rhs_widen: CommonLogicalType<U, V> = rhs;
    return __builtin_bitwise_xor(lhs_widen, rhs_widen);
}

@operator [L, A] << : (value: L, shift_amount: A) -> TypeIf<L, Both<IsLogical<L>, IsIntegral<A>>> = {
    let ushift : L = as_logical(shift_amount);
    return __builtin_shift_left(value, ushift);
}

@operator [L, A] >> : (value: L, shift_amount: A) -> TypeIf<L, Both<IsLogical<L>, IsIntegral<A>>> = {
    let ushift : L = as_logical(shift_amount);
    return __builtin_shift_right(value, ushift);
}

@operator [U, V] == : (lhs: U, rhs: V) -> LogicalCompareResult<U, V> = {
    let lhs_widen: CommonLogicalType<U, V> = lhs;
    let rhs_widen: CommonLogicalType<U, V> = rhs;
    return __builtin_is_equal(lhs_widen, rhs_widen);
}

@operator [U, V] != : (lhs: U, rhs: V) -> LogicalCompareResult<U, V> = {
    return !(lhs == rhs);
}

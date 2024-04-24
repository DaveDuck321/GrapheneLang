@require_once "assignments.c3"
@require_once "logical.c3"
@require_once "type_traits.c3"

typedef [U, V] AreCompatibleArithmeticTypes :
    Either<
        Both<IsIntegral<U>, IsIntegral<V>>,
        Both<IsFloating<U>, IsFloating<V>>,
    >

typedef [U, V] CommonArithmeticType: TypeIf<U, AreCompatibleArithmeticTypes<U, V>>

typedef CommonArithmeticType<i8, i16>   : i16
typedef CommonArithmeticType<i8, i32>   : i32
typedef CommonArithmeticType<i8, i64>   : i64
typedef CommonArithmeticType<i8, i128>  : i128
typedef CommonArithmeticType<i16, i32>  : i32
typedef CommonArithmeticType<i16, i64>  : i64
typedef CommonArithmeticType<i16, i128> : i128
typedef CommonArithmeticType<i32, i64>  : i64
typedef CommonArithmeticType<i32, i128> : i128
typedef CommonArithmeticType<i64, i128> : i128

typedef CommonArithmeticType<f16, f32>  : f32
typedef CommonArithmeticType<f16, f64>  : f64
typedef CommonArithmeticType<f16, f128> : f128
typedef CommonArithmeticType<f32, f64>  : f64
typedef CommonArithmeticType<f32, f128> : f128
typedef CommonArithmeticType<f64, f128> : f128

typedef [U, V] ArithmeticCompareResult : TypeIf<bool, AreCompatibleArithmeticTypes<U, V>>
typedef [T] RegisterHasTrivialAssignments<Override<T>> : TypeIf<TrueType, IsArithmetic<T>>

@operator [T] + : (value: T) -> TypeIf<T, IsArithmetic<T>> = { return value; }

@operator [T] - : (value: T) -> TypeIf<T, IsIntegral<T>> = {
    return __builtin_minus(value);
}

@operator [T] - : (value: T) -> TypeIf<T, IsFloating<T>> = {
    return __builtin_fminus(value);
}

@operator [U, V] + : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_add(lhs_widen, rhs_widen);
}

@operator [U, V] - : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_subtract(lhs_widen, rhs_widen);
}

@operator [U, V] * : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_multiply(lhs_widen, rhs_widen);
}

@operator [U, V] / : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_divide(lhs_widen, rhs_widen);
}

@operator [U, V] % : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_remainder(lhs_widen, rhs_widen);
}

@operator [U, V] == : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_is_equal(lhs_widen, rhs_widen);
}

@operator [U, V] != : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    return !(lhs == rhs);
}

@operator [U, V] < : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_is_less_than(lhs_widen, rhs_widen);
}

@operator [U, V] > : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_is_greater_than(lhs_widen, rhs_widen);
}

@operator [U, V] <= : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    return !(lhs > rhs);
}

@operator [U, V] >= : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    return !(lhs < rhs);
}

typedef[T] HasSpaceship : FalseType

@operator [T] == : (lhs: T, rhs: T) -> TypeIf<bool, HasSpaceship<T>> = {
    return (lhs <=> rhs) == 0;
}

@operator [T] != : (lhs: T, rhs: T) -> TypeIf<bool, HasSpaceship<T>> = {
    return (lhs <=> rhs) != 0;
}

@operator [T] < : (lhs: T, rhs: T) -> TypeIf<bool, HasSpaceship<T>> = {
    return (lhs <=> rhs) < 0;
}

@operator [T] > : (lhs: T, rhs: T) -> TypeIf<bool, HasSpaceship<T>> = {
    return (lhs <=> rhs) > 0;
}

@operator [T] <= : (lhs: T, rhs: T) -> TypeIf<bool, HasSpaceship<T>> = {
    return (lhs <=> rhs) <= 0;
}

@operator [T] >= : (lhs: T, rhs: T) -> TypeIf<bool, HasSpaceship<T>> = {
    return (lhs <=> rhs) >= 0;
}

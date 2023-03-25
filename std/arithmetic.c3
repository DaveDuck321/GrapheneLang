@require_once "type_traits.c3";

typedef[T] IsIntegral : FalseType;
typedef IsIntegral<i8> : TrueType;
typedef IsIntegral<i16> : TrueType;
typedef IsIntegral<i32> : TrueType;
typedef IsIntegral<i64> : TrueType;
typedef IsIntegral<i128> : TrueType;

typedef [U, V] CommonArithmeticType: TypeIf<U, Both<IsIntegral<U>, IsIntegral<V>>>;
typedef CommonArithmeticType<i8, i16> : i16;
typedef CommonArithmeticType<i8, i32> : i32;
typedef CommonArithmeticType<i8, i64> : i64;
typedef CommonArithmeticType<i8, i128> : i128;
typedef CommonArithmeticType<i16, i32> : i32;
typedef CommonArithmeticType<i16, i64> : i64;
typedef CommonArithmeticType<i16, i128> : i128;
typedef CommonArithmeticType<i32, i64> : i64;
typedef CommonArithmeticType<i32, i128> : i128;
typedef CommonArithmeticType<i64, i128> : i128;

typedef [U, V] ArithmeticCompareResult : TypeIf<bool, Both<IsIntegral<U>, IsIntegral<V>>>;

@operator [U, V] + : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_add(lhs_widen, rhs_widen);
};

@operator [U, V] - : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_sub(lhs_widen, rhs_widen);
};

@operator [U, V] * : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_mul(lhs_widen, rhs_widen);
};

@operator [U, V] / : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_div(lhs_widen, rhs_widen);
};

@operator [U, V] % : (lhs: U, rhs : V) -> CommonArithmeticType<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_mod(lhs_widen, rhs_widen);
};

@operator [U, V] == : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_is_equal(lhs_widen, rhs_widen);
};

@operator [U, V] != : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    if (lhs == rhs) {
        return false;
    }
    return true;
};

@operator [U, V] < : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_is_less_than(lhs_widen, rhs_widen);
};

@operator [U, V] > : (lhs: U, rhs: V) -> ArithmeticCompareResult<U, V> = {
    let lhs_widen: CommonArithmeticType<U, V> = lhs;
    let rhs_widen: CommonArithmeticType<U, V> = rhs;
    return __builtin_is_greater_than(lhs_widen, rhs_widen);
};

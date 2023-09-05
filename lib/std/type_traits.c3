typedef TrueType : {}
typedef FalseType : {}
typedef EnableIf<TrueType> : {}

typedef Both<TrueType,  TrueType>  : TrueType
typedef Both<TrueType,  FalseType> : FalseType
typedef Both<FalseType, TrueType>  : FalseType
typedef Both<FalseType, FalseType> : FalseType

typedef Either<TrueType,  TrueType>  : TrueType
typedef Either<TrueType,  FalseType> : TrueType
typedef Either<FalseType, TrueType>  : TrueType
typedef Either<FalseType, FalseType> : FalseType

typedef Not<TrueType> : FalseType
typedef Not<FalseType> : TrueType

typedef[U, V] SelectFirst : U
typedef[Type, Condition] TypeIf : SelectFirst<Type, EnableIf<Condition>>

// Groupings for the builtin types
typedef[T] IsIntegral : FalseType
typedef IsIntegral<i8> : TrueType
typedef IsIntegral<i16> : TrueType
typedef IsIntegral<i32> : TrueType
typedef IsIntegral<i64> : TrueType
typedef IsIntegral<i128> : TrueType

typedef[T] IsLogical : FalseType
typedef IsLogical<bool> : TrueType
typedef IsLogical<u8> : TrueType
typedef IsLogical<u16> : TrueType
typedef IsLogical<u32> : TrueType
typedef IsLogical<u64> : TrueType
typedef IsLogical<u128> : TrueType

typedef[T] HasTrivialAssignments : FalseType

// Needs to implement `flush`, `close`, `consume_length` and `request_buffer`
typedef[T] IsStreamableTextIO : FalseType

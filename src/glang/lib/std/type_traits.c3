typedef TrueType : {}
typedef FalseType : {}
typedef EnableIf<TrueType> : {}

typedef [T] Override : {}

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

typedef [U, V] AreEquivalent : FalseType
typedef [V] AreEquivalent<V, V> : TrueType

typedef [U, V] SelectFirst : U
typedef [Type, Condition] TypeIf : SelectFirst<Type, EnableIf<Condition>>

typedef [T] Ref : T&
typedef [T] Ref<T&> : T&
typedef [T] Ref<T mut&> : T&

typedef [T] MutRef : TypeIf<T mut&, Not<IsRef<T>>>
typedef [T] MutRef<T mut&> : T mut&

typedef [T] IsRef : Either<IsMutRef<T>, IsConstRef<T>>

typedef [T] IsMutRef : FalseType
typedef [T] IsMutRef<T mut&> : TrueType
typedef [T] IsMutRef<T[mut &]> : TrueType

typedef [T] IsConstRef : FalseType
typedef [T] IsConstRef<T&> : TrueType
typedef [T] IsConstRef<T[&]> : TrueType

// Groupings for the builtin types
typedef [T] IsIntegral : FalseType
typedef IsIntegral<i8> : TrueType
typedef IsIntegral<i16> : TrueType
typedef IsIntegral<i32> : TrueType
typedef IsIntegral<i64> : TrueType
typedef IsIntegral<i128> : TrueType

typedef [T] IsFloating : FalseType
typedef IsFloating<f16> : TrueType
typedef IsFloating<f32> : TrueType
typedef IsFloating<f64> : TrueType
typedef IsFloating<f128> : TrueType

typedef [T] IsArithmetic : Either<IsIntegral<T>, IsFloating<T>>

typedef [T] IsLogical : FalseType
typedef IsLogical<bool> : TrueType
typedef IsLogical<u8> : TrueType
typedef IsLogical<u16> : TrueType
typedef IsLogical<u32> : TrueType
typedef IsLogical<u64> : TrueType
typedef IsLogical<u128> : TrueType

typedef [T] HasTrivialAssignments : RegisterHasTrivialAssignments<Override<T>>
typedef [T] RegisterHasTrivialAssignments : FalseType

// Needs to implement `flush`, `close`, `consume_length` and `request_buffer`
typedef [T] IsStreamableTextIO : RegisterIsStreamableTextIO<Override<T>>
typedef [T] RegisterIsStreamableTextIO : FalseType

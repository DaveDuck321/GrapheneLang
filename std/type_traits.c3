typedef TrueType : {};
typedef FalseType : {};
typedef EnableIf<TrueType> : {};

typedef Both<TrueType,  TrueType>  : TrueType;
typedef Both<TrueType,  FalseType> : FalseType;
typedef Both<FalseType, TrueType>  : FalseType;
typedef Both<FalseType, FalseType> : FalseType;

typedef Either<TrueType,  TrueType>  : TrueType;
typedef Either<TrueType,  FalseType> : TrueType;
typedef Either<FalseType, TrueType>  : TrueType;
typedef Either<FalseType, FalseType> : FalseType;

typedef Not<TrueType> : FalseType;
typedef Not<FalseType> : TrueType;

typedef[U, V] SelectFirst : U;
typedef[Type, Condition] TypeIf : SelectFirst<Type, EnableIf<Condition>>;

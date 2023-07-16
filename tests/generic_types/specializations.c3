typedef TrueType : {}
typedef FalseType : {}

typedef[T] IsIntegral : FalseType
typedef IsIntegral<int> : TrueType
typedef IsIntegral<u8> : TrueType
typedef IsIntegral<u16> : TrueType

foreign puts : (str : u8[&]) -> int

function to_value : (_: TrueType) -> bool = {
    return true;
}

function to_value : (_: FalseType) -> bool = {
    return false;
}

function main : () -> int = {
    let result1 : IsIntegral<u16> = {};
    if to_value(result1){
        puts("u16 is integral\0");
    }

    let result2 : IsIntegral<u8[&]> = {};
    if to_value(result2){
        // Should not run
        puts("u8[&] is integral\0");
    }
    return 0;
}

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// u16 is integral

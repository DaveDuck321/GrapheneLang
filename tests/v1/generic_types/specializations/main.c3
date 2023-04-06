@require_once "std/io.c3";

typedef TrueType : {};
typedef FalseType : {};

typedef[T] IsIntegral : FalseType;
typedef IsIntegral<int> : TrueType;
typedef IsIntegral<u8> : TrueType;
typedef IsIntegral<u16> : TrueType;

function to_value : (_: TrueType) -> bool = {
    return true;
};

function to_value : (_: FalseType) -> bool = {
    return false;
};

function main : () -> int = {
    let result1 : IsIntegral<u16> = {};
    if (to_value(result1)) {
        puts("u16 is integral");
    }

    let result2 : IsIntegral<u8[&]> = {};
    if (to_value(result2)) {
        // Should not run
        puts("u8[&] is integral");
    }
    return 0;
};

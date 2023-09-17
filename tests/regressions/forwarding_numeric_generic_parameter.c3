typedef[@Len] S : int[@Len]
typedef[@Len] T1 : S<@Len>


function main : () -> int = {
    mut a : T1<10>;
    a[8] = 6;
    return a[8];
}

/// @COMPILE
/// @RUN; EXPECT 6

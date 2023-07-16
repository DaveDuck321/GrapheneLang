foreign puts: (str: u8[&]) -> int

function main: () -> int = {
    // There are two characters between the β and the γ:
    // - https://unicode-table.com/en/00AD/ (0xc2ad in utf-8)
    // - https://unicode-table.com/en/2000/ (0xe28080 in utf-8)
    puts("άβ­ γ");
    return 0;
}

/// @COMPILE --use-crt
/// @GREP_IR [11 x i8] c"άβ\c2\ad\e2\80\80γ"
/// @RUN; EXPECT OUT
/// άβ­ γ

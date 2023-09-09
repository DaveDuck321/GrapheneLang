foreign gnu_get_libc_version : () -> u8[&]

foreign puts : (str: u8[&]) -> int

function main : () -> int = {
    puts(&gnu_get_libc_version());
    return 0;
}

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// 2.*

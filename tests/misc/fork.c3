@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/unistd.c3"

function main : () -> int = {
    let pid : pid_t = fork();

    if pid != 0 {
        puts<6>("Parent");
    } else {
        puts<5>("Child");
    }

    return 0;
}

/// @FOR x86_64_linux
/// @COMPILE
/// @RUN; EXPECT OUT
/// Parent
/// Child

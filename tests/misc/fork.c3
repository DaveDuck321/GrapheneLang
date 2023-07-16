@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/unistd.c3"

function main : () -> int = {
    const pid : pid_t = fork();

    if (pid != 0) {
        puts<6>("Parent");
    } else {
        puts<5>("Child");
    }

    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// Parent
/// Child

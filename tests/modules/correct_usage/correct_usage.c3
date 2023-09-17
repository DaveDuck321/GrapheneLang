@require_once "subdir/_thing.c3"
@require_once "_stuff.c3"
@require_once "_stuff.c3"
@require_once "_stuff.c3"

function main : () -> int = {
    mut a : SubDirStruct;

    return __builtin_add(do_thing(), do_stuff());
}

/// @COMPILE -Iincluded
/// @RUN; EXPECT 9

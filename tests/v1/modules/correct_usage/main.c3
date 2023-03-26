@require_once "subdir/thing.c3";
@require_once "stuff.c3";
@require_once "stuff.c3";
@require_once "stuff.c3";

function main : () -> int = {
    let a : SubDirStruct;

    return __builtin_add(do_thing(), do_stuff());
};

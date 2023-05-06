@require_once "_other_thing.c3";

typedef SubDirStruct : {item : int};

function do_thing : () -> int = {
    return __builtin_add(do_other_thing(), 1);
};

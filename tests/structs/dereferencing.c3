typedef MyStruct : {member : int};

foreign GetStruct : () -> MyStruct;

function fn_val : () -> MyStruct = {
    return GetStruct();
};

function fn_ref : (in : MyStruct&) -> MyStruct& = {
    return &in;
};


function main: () -> int = {
    let a : MyStruct = GetStruct();
    let b : MyStruct& = &a;

    let result0 : int = a.member;
    let result1 : int = fn_val().member;

    let result2 : int = b.member;
    let result3 : int = fn_ref(&a).member;
    let result4 : int = fn_ref(&b).member;

    return 0;
};

/// @COMPILE

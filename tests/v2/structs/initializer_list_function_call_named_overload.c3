foreign puts : (str : u8[&]) -> void;

typedef MyStruct1 : {a: int, z: int};
typedef MyStruct2 : {a: int, y: int};
typedef MyStruct3 : {a: int, x: int};

function get : (struct : MyStruct1) -> int = {
    puts("MyStruct1");
    return struct.z;
};

function get : (struct : MyStruct2) -> int = {
    puts("MyStruct2");
    return struct.y;
};

function get : (struct : MyStruct3) -> int = {
    puts("MyStruct3");
    return struct.x;
};


function main : () -> int = {
    get({a : 11, y : 12});
    return 0;
};

/// @COMPILE --use-crt
/// @RUN; EXPECT OUT
/// MyStruct2

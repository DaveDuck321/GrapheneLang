typedef my_new_type : int;
typedef my_new_type_2 : my_new_type;

function b : (input : my_new_type_2) -> int = {
    return __builtin_add(2, input);
};

function main: () -> int = {
    return b(6);
};

/// @COMPILE
/// @RUN; EXPECT 8

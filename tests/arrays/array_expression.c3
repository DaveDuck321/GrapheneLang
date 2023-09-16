function[T] generic_fn : () -> void = {

}

function main : () -> int = {
    mut array : int[6];
    array[1] = 1;
    generic_fn<int[6]>();
    return array[1];
}


/// @COMPILE
/// @RUN; EXPECT 1

typedef my_void : void;
typedef my_struct : { a: int, b: my_void };

function main : () -> int = {
    let s : my_struct;

    return 0;
};

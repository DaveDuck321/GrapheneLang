typedef MyStruct : {a : int};

foreign puts : (str : u8[&]) -> int;

function fn : (a: int) -> int = {
    puts("val");
    return 0;
};

function fn : (a: int&) -> int = {
    puts("ref");
    return 0;
};

function main: () -> int = {
    let x: int = 0;
    let my_struct : MyStruct; // TODO: initialize (and warn when uninitialized)
    fn(x);
    fn(&x);
    fn(0);
    fn(my_struct.a);
    fn(&my_struct.a);
    return 0;
};

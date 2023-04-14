@require_once "std/arithmetic.c3";
@require_once "std/memory.c3";


typedef MyStruct : {
    a : int,
    b : isize
};


function main : () -> int = {
    let allocator : Allocator = initialize_allocator();

    let a : int& = &allocator:allocate<int>();
    let b : MyStruct& = &allocator:allocate<MyStruct>();

    let c : i64[100]& = &allocator:allocate<i64[100]>();
    let d : i64[800]& = &allocator:allocate<i64[800]>();
    allocator:deallocate<i64[100]>(&c);

    let e : MyStruct& = &allocator:allocate<MyStruct>();

    a = 10;
    b.a = 27;
    b.b = 5;

    if (a + b.a + b.b == 42) {
        return 0;
    }

    return 1;
};

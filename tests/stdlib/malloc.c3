@require_once "std/arithmetic.c3"
@require_once "std/memory.c3"
@require_once "std/wrappers.c3"
@require_once "std/array.c3"

typedef MyStruct : {
    a : int,
    b : isize,
    c : isize,
    d : bool,
    f : isize,
}

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();

    mut allocated_memory: Ptr<int>[1028];
    mut allocated_structs: Ptr<MyStruct>[512];

    // Allocate int array
    mut i: int = 0;
    while i < 1028 {
        let allocated : int mut& = &mut allocator:allocate<int>();
        allocated_memory[i] = {&allocated};
        allocated = i;
        i = i + 1;
    }

    // Check int array contains valid data
    i = 0;
    while i < 1028 {
        if allocated_memory[i]:data() != i {
            return 1;
        }
        i = i + 1;
    }

    // Deallocate every other element of int array
    i = 0;
    while i < 1028 {
        allocator:deallocate(&allocated_memory[i]:data());
        i = i + 2;
    }

    // Allocate an unrelated struct array
    i = 0;
    while i < 512 {
        let allocated : MyStruct mut& = &mut allocator:allocate<MyStruct>();
        allocated_structs[i] = {&allocated};
        allocated = {-1, -1, 0, false, 89};
        i = i + 1;
    }

    // Deallocate the struct array
    i = 0;
    while i < 512 {
        allocator:deallocate(&allocated_structs[i]:data());
        i = i + 1;
    }

    // Check the remaining elements of the int array are still valid
    i = 1;
    while i < 1028 {
        if allocated_memory[i]:data() != i {
            return 1;
        }
        allocator:deallocate(&allocated_memory[i]:data());
        i = i + 2;
    }

    // Allocate the struct array again to flush out any safety issues
    i = 0;
    while i < 512 {
        let allocated : MyStruct mut& = &mut allocator:allocate<MyStruct>();
        allocated_structs[i] = {&allocated};
        allocated = {-1, -1, 0, false, 7};
        i = i + 1;
    }

    return 0;
}

/// @COMPILE
/// @RUN

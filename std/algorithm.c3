@require_once "arithmetic.c3";
@require_once "logical.c3";

function heap_left_child: (i: isize) -> isize = {
    return 2 * i + 1;
};

function heap_right_child: (i: isize) -> isize = {
    return 2 * i + 2;
};

function heap_parent: (i: isize) -> isize = {
    return (i - 1) / 2;
};

function[T] swap: (x: T&, y: T&) -> void = {
    const temp : T = x;
    x = y;
    y = temp;
};

function[T] max_heapify: (array: T[&], heap_size: isize, i: isize) -> void = {
    // runtime_assert(i >= 0);
    // runtime_assert(heap_size > i);

    const left_i : isize = heap_left_child(i);
    const right_i : isize = heap_right_child(i);

    let max_i : isize = i;

    if (heap_size > left_i && array[left_i] > array[max_i]) {
        max_i = left_i;
    }

    if (heap_size > right_i && array[right_i] > array[max_i]) {
        max_i = right_i;
    }

    if (max_i != i) {
        swap<T>(&array[i], &array[max_i]);
        max_heapify<T>(&array, max_i);
    }
};

function[T] build_max_heap: (array: T[&], heap_size: isize) -> void = {
    let i : isize = heap_size / 2 - 1;

    // TODO: use a for loop...
    while (i >= 0) {
        max_heapify<T>(array, heap_size, i);
        i = i - 1;
    }
};

function[T] heap_sort: (array: T[&], heap_size: isize) -> void = {
    build_max_heap<T>(array);

    let heap_size : isize = Len;
    while (heap_size > 1) {
        swap<T>(&array[heap_size - 1], &array[0]);
        heap_size = heap_size - 1;

        max_heapify<T>(array, heap_size, 0);
    }
};

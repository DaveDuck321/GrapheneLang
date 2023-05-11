@require_once "std/algorithm.c3";
@require_once "std/arithmetic.c3";
@require_once "std/io.c3";

function print_num : (num: int) -> void = {
    let strs: u8[&][10] = {
        "0\n",
        "1\n",
        "2\n",
        "3\n",
        "4\n",
        "5\n",
        "6\n",
        "7\n",
        "8\n",
        "9\n"
    };

    sys_write(1, &strs[num], 2);
};

function main : () -> int = {
    let array    : int[7];
    let expected : int[7];

    let i : int = 0;
    while (i < 7) {
        array[i] = 7 - i;
        expected[i] = i;

        i = i + 1;
    }

    heap_sort<int>(&array, 7);
    max_heapify<int>(&array, 7, 0);

    i = 0;
    while (i < 7) {
        if (array[i] != expected[i]) {
            print_num(i);
            print_num(array[i]);
            print_num(expected[i]);
            return array[i];
        }

        i = i + 1;
    }

    return 0;
};

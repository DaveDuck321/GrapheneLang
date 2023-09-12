@require_once "std/arithmetic.c3";
@require_once "std/memory.c3";
@require_once "std/io.c3";
@require_once "syscalls.c3";

function[T] pack_len: (curr: T) -> int = {
    // Base case
    return 1;
}

function[T, Ys...] pack_len: (curr: T, next: Ys...) -> int = {
    // Very Haskell
    return pack_len(next...) + 1;
}

function to_str: (allocator: Allocator&, val: int) -> iovec = {
    // FIXME should be const.
    let strs: u8[&][10] = {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    };

    let iov: iovec = {&strs[val], 1};

    // TODO numbers > 10 and < 0.
    return iov;
}

function[@Len] to_str: (allocator: Allocator&, val: u8[@Len]) -> iovec = {
    // FIXME we leak memory here.
    let mem: u8[@Len]& = &allocator:allocate<u8[@Len]>();

    let i: isize = 0;
    while i < @Len {
        mem[i] = val[i];
        i = i + 1;
    }

    return {&mem, @Len};
}

function[T] printf_single: (allocator: Allocator&, iov: iovec[&], i: isize, val: T) -> void = {
    iov[i] = to_str(&allocator, val);
}

function[@Len, T] printf_impl: (allocator: Allocator&,
                                       format: u8[@Len]&,
                                       format_i: isize,
                                       iov: iovec[&],
                                       iov_i: isize,
                                       val: T
                                      ) -> int = {
    let format_j : isize = format_i;
    while format_j < @Len {
        // 35 is # in ASCII.
        if format[format_j] == __builtin_bitcast<u8>(__builtin_narrow<i8>(35) {
            // FIXME memory leak.
            let mem: u8[@Len]& = &allocator:allocate<u8[@Len]>();
            let i: isize = format_i;
            while i < format_j {
                mem[i] = format[i];
                i = i + 1;
            }

            iov[iov_i] = {&mem, format_j - format_i};
            printf_single(&allocator, &iov, i+1, val);

            return 2;
        }

        format_j = format_j + 1;
    }

    let mem: u8[@Len]& = &allocator:allocate<u8[@Len]>();
    let i: isize = format_i;
    while i < format_j {
        mem[i] = format[i];
        i = i + 1;
    }
    iov[iov_i] = {&mem, format_j - format_i};

    return 1;
}

function[@Len, T, Ys...] printf_impl: (allocator: Allocator&,
                                       format: u8[@Len]&,
                                       format_i: isize,
                                       iov: iovec[&],
                                       iov_i: isize,
                                       val: T,
                                       next_vals: Ys...) -> int = {
    let format_j : isize = format_i;
    while format_j < @Len {
        // 35 is # in ASCII.
        if format[format_j] == __builtin_bitcast<u8>(__builtin_narrow<i8>(35) {
            // FIXME memory leak.
            let mem: u8[@Len]& = &allocator:allocate<u8[@Len]>();
            let i: isize = format_i;
            while i < format_j {
                mem[i] = format[i];
                i = i + 1;
            }

            iov[iov_i] = {&mem, format_j - format_i};
            printf_single(&allocator, &iov, i+1, val);

            return 2 + printf_impl(&allocator, &format, format_j+1, &iov, i+2, next_vals...);
        }

        format_j = format_j + 1;
    }

    let mem: u8[@Len]& = &allocator:allocate<u8[@Len]>();
    let i: isize = format_i;
    while i < format_j {
        mem[i] = format[i];
        i = i + 1;
    }
    iov[iov_i] = {&mem, format_j - format_i};

    return 1;
}

function[@Len, Ts...] printf: (format: u8[@Len]&, vals: Ts...) -> void = {
    let allocator: Allocator = initialize_allocator();

    // 32 elements should be enough for everyone :).
    let iov: iovec[32];

    let fixme: iovec[&] = &iov;
    let len: int = printf_impl(&allocator, &format, 0, &fixme, 0, vals...);
    sys_writev(1, &fixme, len);
}

function main : () -> int = {
    // printf("one # #\n", "two", "three");
    printf("a: #, b: #, c: #\n", 1, 2, 3);

    return 0;
}

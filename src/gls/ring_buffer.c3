@require_once "std/unistd.c3"
@require_once "std/sys/mman.c3"
@require_once "std/string.c3"
@require_once "std/wrappers.c3"
@require_once "std/span.c3"

// See https://lo.calho.st/posts/black-magic-buffer/ for an explanation.
typedef RingBuffer : {
    buffer : Span<u8>,
    size : isize,
    head : isize,
    tail : isize,
}

function make<RingBuffer> : (size : isize, name : CString) -> RingBuffer = {
    // TODO get the page size from sysconf().
    runtime_assert(size % 4096 == 0);

    let fd : int = memfd_create(name, 0x0);
    runtime_assert(fd >= 0);

    runtime_assert(ftruncate(fd, size) == 0);

    let addr : iptr = mmap(
        0, 2 * size, SYS_PROT_NONE(), SYS_MAP_PRIVATE() | SYS_MAP_ANONYMOUS(),
        -1, 0
    );

    mmap(
        addr, size, SYS_PROT_READ() | SYS_PROT_WRITE(),
        SYS_MAP_SHARED() | SYS_MAP_FIXED(), fd, 0
    );
    mmap(
        addr + size, size, SYS_PROT_READ() | SYS_PROT_WRITE(),
        SYS_MAP_SHARED() | SYS_MAP_FIXED(), fd, 0
    );

    sys_close(fd);

    return {
        .buffer = { &mut addr_to_mut_heap_array<u8>(addr), 2 * size },
        .size = size,
        .head = 0,
        .tail = 0,
    };
}

function deallocate : (self : RingBuffer mut&) -> void = {
    let addr : iptr = ref_to_addr(&self.buffer.data);

    self:clear();

    runtime_assert(munmap(addr, 2 * self.size) == 0);

    let null : iptr = 0;
    self.buffer = { &mut addr_to_mut_heap_array<u8>(null), 0 };
    self.size = 0;
}

function read_impl : (
    buffer : RingBuffer mut&, source : File
) -> isize = {
    // We need a span over the free space, so it's easier if we just unwrap the
    // head.
    mut head : isize = buffer.head;
    if head <= buffer.tail {
        head += buffer.size;
    }

    let free : Span<u8> = buffer.buffer:slice_between(buffer.tail, head - 1);
    let view : View<u8> = source:read(free);

    buffer.tail += view:length();
    buffer.tail %= buffer.size;

    return view:length();
}

function find_newline : (buffer : RingBuffer mut&) -> Optional<isize> = {
    mut tail : isize = buffer.tail;
    if tail < buffer.head {
        tail += buffer.size;
    }

    for i in range(buffer.head, tail) {
        if buffer.buffer:get(i) == char("\n") {
            return make<Optional<isize>>(i);
        }
    }

    return make<Optional<isize>>();
}

function read_line : (
    buffer : RingBuffer mut&, source : File
) -> Optional<View<u8>> = {
    while true {
        // Find the first '\n'.
        let newline_idx : Optional<isize> = buffer:find_newline();
        if newline_idx:has_value() {
            return make<Optional<View<u8>>>(to_view(buffer.buffer:slice_between(buffer.head, newline_idx:data() + 1)));
        }

        // Read from the fd and try again.
        if buffer:read_impl(source) == 0 {
            // Read failed, give up.
            print(sv("ERROR: read_line - read failed"));
            return make<Optional<View<u8>>>();
        }
    }

    // FIXME compiler bug, this shouldn't be necessary.
    return make<Optional<View<u8>>>();
}

function read : (
    buffer : RingBuffer mut&, source : File, size : isize
) -> Optional<View<u8>> = {
    // Maximum capacity is size - 1.
    if size > buffer.size - 1 {
        return make<Optional<View<u8>>>();
    }

    while (buffer.tail - buffer.head) % buffer.size < size {
        // Read from the fd and try again.
        if buffer:read_impl(source) == 0 {
            // Read failed, give up.
            return make<Optional<View<u8>>>();
        }
    }

    return make<Optional<View<u8>>>(
        to_view(buffer.buffer:slice_between(buffer.head, buffer.head + size))
    );
}

function advance : (buffer : RingBuffer mut&, size : isize) -> void = {
    // TODO bounds checking.
    buffer.head += size;
    buffer.head %= buffer.size;
}

function clear : (buffer : RingBuffer mut&) -> void = {
    // Clear the contents of the buffer.
    buffer.head = 0;
    buffer.tail = 0;
}

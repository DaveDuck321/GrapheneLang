@require_once "algorithms.c3"
@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "memory.c3"
@require_once "wrappers.c3"

typedef [T] Vector : {
    allocator : Allocator&,
    memory : Span<T>,
    length : isize,
}

function DEFAULT_VECTOR_CAPACITY : () -> isize = { return 8; }

function [T] make_vector : (allocator : Allocator&) -> Vector<T> = {
    // HACK we don't want to allocate by default, so we are forced to initialize
    // the Span with a null pointer. Note that if a user calls data() before
    // any memory is allocated, then they'll get a 0-length span with a null
    // pointer.
    return {
        .allocator = &allocator,
        .memory = { &__builtin_int_to_ptr<T[&]>(0), 0 },
        .length = 0,
    };
}

function [T] push_back : (self : Vector<T>&, data : T) -> void = {
    if self.length == self.memory.length {
        // Grow the vector
        let new_capacity : isize = max(2 * self.memory.length, DEFAULT_VECTOR_CAPACITY());
        let new_memory : Span<T> = self.allocator:allocate_span<T>(new_capacity);

        // Copy the old data
        // TODO: use a memcpy/ for loop
        let index : isize = 0;
        while index < self.length {
            new_memory.data[index] = self.memory.data[index];
            index += 1;
        }

        if self.memory.length != 0 {
            // BUG deallocating a null pointer segfaults.
            self.allocator:deallocate_span(self.memory);
        }
        self.memory = new_memory;
    }
    self.memory.data[self.length] = data;

    self.length += 1;
}

function [T] extend_back : (self : Vector<T>&, data: Span<T>) -> void = {
    // TODO: create capacity in one go
    for value in data:get_iter() {
        self:push_back(value);
    }
}

function [T] pop_back : (self : Vector<T>&) -> T = {
    runtime_assert(self.length > 0);

    self.length -= 1;
    return self.memory.data[self.length];
}

function [T] get : (self : Vector<T>&, index : isize) -> T& = {
    runtime_assert(index >= 0 and index < self.length);
    return &self.memory.data[index];
}

function [T] set : (self : Vector<T>&, index : isize, data : T) -> void = {
    runtime_assert(index >= 0 and index < self.length);
    self.memory.data[index] = data;
}

function [T] length : (self : Vector<T>&) -> isize = {
    return self.length;
}

function [T] capacity : (self : Vector<T>&) -> isize = {
    return self.memory.length;
}

function [T] clear : (self : Vector<T>&) -> void = {
    // TODO: destructor support (if we add destructors)
    self.length = 0;
}

function [T] deallocate : (self : Vector<T>&) -> void = {
    // TODO: this aught to be a destructor
    self.allocator:deallocate_span(self.memory);
}

function [T] data : (self : Vector<T>&) -> Span<T> = {
    return {
        .data = &self.memory.data,
        .length = self.length,
    };
}


typedef [T] VectorIter : {
    vector : Vector<T>&,
    index : isize,
}

function [T] get_iter : (vec : Vector<T>&) -> VectorIter<T> = {
    return {&vec, 0};
}

@implicit [T] has_next : (iter : VectorIter<T>&) -> bool = {
    return iter.index < iter.vector.length;
}

@implicit [T] get_next : (iter : VectorIter<T>&) -> T& = {
    let index : isize = iter.index;
    iter.index += 1;
    return &iter.vector:get(index);
}

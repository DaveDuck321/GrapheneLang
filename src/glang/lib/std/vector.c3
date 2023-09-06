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
    /* TODO: It'd be kinda nice to NOT allocate by default
             atm it would be hard to make `Span` support this */
    return {
        .allocator = &allocator,
        .memory = allocator:allocate_span<T>(DEFAULT_VECTOR_CAPACITY()),
        .length = 0,
    };
}

function [T] push_back : (self : Vector<T>&, data : T) -> void = {
    if self.length == self.memory.length {
        // Grow the vector
        let new_capacity : isize = 2 * self.memory.length;
        let new_memory : Span<T> = self.allocator:allocate_span<T>(new_capacity);

        // Copy the old data
        // TODO: use a memcpy/ for loop
        let index : isize = 0;
        while index < self.length {
            new_memory.data[index] = self.memory.data[index];
            index += 1;
        }

        self.allocator:deallocate_span(self.memory);
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

@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "iterators.c3"
@require_once "logical.c3"
@require_once "sys/mman.c3"
@require_once "util.c3"
@require_once "wrappers.c3"

function allocate_page_and_return_address : (length : isize) -> iptr = {
    let prot : u32 = SYS_PROT_READ() | SYS_PROT_WRITE();
    let flags : u32 = SYS_MAP_PRIVATE() | SYS_MAP_ANONYMOUS();

    return mmap(0, length, prot, flags, -1, 0);
}

function deallocate_page : (addr: iptr, length : isize) -> void = {
    munmap(addr, length);
}

function PAGE_SIZE : () -> isize = {
    return 4096;  // TODO: this should be a compile time constant
}

typedef AllocatorNode : {
    integrity_check : u64,
    next_node : iptr, // TODO use: Optional<Ptr<AllocatorNode>> (recursive type)
}

typedef Allocator : {
    // TODO: parameterize `8` as a generic argument
    integrity_checks : u64[8],
    bins : Optional<Ptr<AllocatorNode>>[8],
}

function get_allocator_bin_index : (size_required : isize) -> isize = {
    let bin_index : int = 0;
    let size_over_current_bin : isize = size_required / sizeof<AllocatorNode>();

    while size_over_current_bin != 0 {
        bin_index += 1;
        // TODO: maybe we should have signed bitshift... but also LLVM can defo optimize the division
        size_over_current_bin = size_over_current_bin / 2;
    }

    return bin_index;
}

function get_block_size_from_bin_index : (bin : isize) -> isize = {
    let bin0_size: u64 = as_logical(sizeof<AllocatorNode>());
    return as_arithmetic(bin0_size << bin);
}

function allocate_impl : (allocator : Allocator&, requested_size : isize) -> isize = {
    let bin_index : isize = get_allocator_bin_index(requested_size);

    if bin_index >= allocator.bins:length() {
        // We just pass large allocations straight to the kernel
        return allocate_page_and_return_address(requested_size);
    } else {
        // Otherwise we actually do it ourselves
        return allocate_from_bin_and_return_address(&allocator, bin_index);
    }
}


function [T] allocate : (allocator : Allocator&) -> T& = {
    return &addr_to_ref<T>(allocator:allocate_impl(sizeof<T>()));
}

function [T] allocate_span : (allocator : Allocator&, length: isize) -> Span<T> = {
    return {
        .data = &addr_to_heap_array<T>(allocator:allocate_impl(sizeof<T>() * length)),
        .length = length,
    };
}


function deallocate_impl : (allocator : Allocator&, memory_address: isize, size : isize) -> void = {
    let bin_index : isize = get_allocator_bin_index(size);
    if bin_index >= allocator.bins:length() {
        deallocate_page(memory_address, size);
    } else {
        deallocate_to_bin(&allocator, bin_index, memory_address);
    }
}

function [T] deallocate : (allocator : Allocator&, ref : T&) -> void = {
    allocator:deallocate_impl(ref_to_addr(&ref), sizeof<T>());
}

function [T] deallocate_span : (allocator : Allocator&, span : Span<T>) -> void = {
    allocator:deallocate_impl(ref_to_addr(&span.data), sizeof<T>() * span.length);
}

function deallocate_to_bin : (allocator : Allocator&, bin : isize, addr : iptr) -> void = {
    let integrity_check : u64 = allocator.integrity_checks[bin];

    let new_node : AllocatorNode& = &addr_to_ref<AllocatorNode>(addr);
    new_node.integrity_check = integrity_check;

    if allocator.bins[bin]:has_value() {
        let next_node_ptr : Ptr<AllocatorNode> = allocator.bins[bin]:data();
        let next_node : AllocatorNode& = &next_node_ptr:data();
        runtime_assert(next_node.integrity_check == integrity_check);

        new_node.next_node = ref_to_addr(&next_node);
    } else {
        new_node.next_node = 0;
    }
    allocator.bins[bin]:store<Ptr<AllocatorNode>>({&new_node});
}

function allocate_from_bin_and_return_address : (allocator : Allocator&, bin : isize) -> iptr = {
    // No bins available, lets allocate more memory
    if bin == allocator.bins:length() {
        runtime_assert(get_block_size_from_bin_index(bin) == PAGE_SIZE());
        return allocate_page_and_return_address(PAGE_SIZE());
    }

    // Normal happy path
    let integrity_check : u64 = allocator.integrity_checks[bin];

    // TODO: need template arguments deduction for function calls (especially the UFCS)
    if allocator.bins[bin]:has_value() {
        let node_to_pop_ptr : Ptr<AllocatorNode> = allocator.bins[bin]:data();
        let node_to_pop : AllocatorNode& = &node_to_pop_ptr:data();

        runtime_assert(node_to_pop.integrity_check == integrity_check);
        // TODO: we should eventually just copy the optional without the branch
        if node_to_pop.next_node != 0 {
            let next_node_but_fixme_address : iptr = node_to_pop.next_node;
            let next_node : AllocatorNode& = &addr_to_ref<AllocatorNode>(next_node_but_fixme_address);
            let next_node_ptr : Ptr<AllocatorNode> = {&next_node};

            allocator.bins[bin]:store(next_node_ptr);
        } else {
            allocator.bins[bin]:erase();
        }

        return ref_to_addr(&node_to_pop);
    } else {
        // We don't have any bins left, split one from the bin above us
        let this_size : isize = get_block_size_from_bin_index(bin);

        let new_bin_location_1 : iptr  = allocate_from_bin_and_return_address(&allocator, bin + 1);
        let new_bin_location_2 : iptr = new_bin_location_1 + this_size;

        // We give the memory in bin1 to the user immediately, but we save bin2 for ourselves
        let new_node : AllocatorNode& = &addr_to_ref<AllocatorNode>(new_bin_location_2);
        new_node.integrity_check = integrity_check;
        new_node.next_node = 0;
        allocator.bins[bin]:store<Ptr<AllocatorNode>>({&new_node});

        return new_bin_location_1;
    }
}

function initialize_allocator : () -> Allocator = {
    // TODO: should actually check that the bin is a power of 2 bytes
    runtime_assert(sizeof<AllocatorNode>() == 16);

    let allocator : Allocator;

    for bin in range(allocator.bins:length()) {
        // TODO: use a true random number
        // TODO: allow 64 bit constants
        allocator.integrity_checks[bin] = 0xd2f4bd5b;
        allocator.bins[bin]:erase();
    }
    return allocator;
}

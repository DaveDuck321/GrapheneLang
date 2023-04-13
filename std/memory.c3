@require_once "arithmetic.c3";
@require_once "arrays.c3";
@require_once "logical.c3";
@require_once "syscalls.c3";
@require_once "util.c3";
@require_once "wrappers.c3";

function allocate_page_and_return_address : (length : isize) -> isize = {
    let prot : u32 = SYS_PROT_READ() | SYS_PROT_WRITE();
    let flags : u32 = SYS_MAP_PRIVATE() | SYS_MAP_ANONYMOUS();

    return sys_mmap(0, length, prot, flags, -1, 0);
};

function PAGE_SIZE : () -> isize = {
    return 4096;  // TODO: this should be a compile time constant
};

typedef AllocatorNode : {
    integrity_check : u64,
    next_node : isize // TODO use: Optional<Ptr<AllocatorNode>> (recursive type)
};

typedef Allocator : {
    // TODO: parameterize `9` as a generic argument
    integrity_checks : u64[9],
    bins : Optional<Ptr<AllocatorNode>>[9]
};

function get_allocator_bin_index : (size_required : isize) -> isize = {
    let bin_index : int = 0;
    let size_over_current_bin : isize = size_required / 8;  // TODO: bin0 is 8 bytes (this should be cleaner)

    while (size_over_current_bin != 0) {
        bin_index = bin_index + 1;
        // TODO: maybe we should have signed bitshift... but also LLVM can defo optimize the division
        size_over_current_bin = size_over_current_bin / 2;
    }

    runtime_assert(9 > bin_index);
    return bin_index;
};

function get_block_size_from_bin_index : (bin : isize) -> isize = {
    let bin0_size: u64 = 0x8;
    // TODO: this signed/ unsigned conversion is ugly
    return __builtin_bitcast<isize>(bin0_size << __builtin_bitcast<u64>(bin));
};

function [T] allocate : (allocator : Allocator&) -> T& = {
    let memory_address : isize;
    if (sizeof<T>() > PAGE_SIZE()) {
        // We just pass large allocations straight to the kernel
        memory_address = allocate_page_and_return_address(sizeof<T>());
    } else {
        // Otherwise we actually do it ourselves
        let bin_index : isize = get_allocator_bin_index(sizeof<T>());
        memory_address = allocate_from_bin_and_return_address(&allocator, bin_index);
    }

    return &__builtin_int_to_ptr<T&>(memory_address);
};

function allocate_from_bin_and_return_address : (allocator : Allocator&, bin : isize) -> isize = {
    // No bins available, lets allocate more memory
    // TODO: hacky magic number
    if (bin == 9) {
        return allocate_page_and_return_address(PAGE_SIZE());
    }

    // Normal happy path
    let integrity_check : u64 = allocator.integrity_checks[bin];

    // FIXME: lots of broken UFCS
    // TODO: need template arguments deduction for function calls
    if (has_value<Ptr<AllocatorNode>>(&allocator.bins[bin])) {
        let node_to_pop_ptr : Ptr<AllocatorNode> = data<Ptr<AllocatorNode>>(&allocator.bins[bin]);
        let node_to_pop : AllocatorNode& = &data<AllocatorNode>(&node_to_pop_ptr);

        runtime_assert(node_to_pop.integrity_check == integrity_check);

        // TODO: we should eventually just copy the optional without the branch
        if (node_to_pop.next_node != 0) {
            let next_node_but_fixme_address : isize = node_to_pop.next_node;
            let next_node : Ptr<AllocatorNode>& = &__builtin_int_to_ptr<Ptr<AllocatorNode>&>(next_node_but_fixme_address);

            allocator.bins[bin] = make_optional<Ptr<AllocatorNode>>(next_node);
        } else {
            // TODO: this deduction should work
            allocator.bins[bin] = make_optional<Ptr<AllocatorNode>>();
        }

        return __builtin_ptr_to_int<isize>(node_to_pop);

    } else {
        // We don't have any bins left, split one from the bin above us
        let this_size : isize = get_block_size_from_bin_index(bin);

        let new_bin_location_1 : isize = allocate_from_bin_and_return_address(&allocator, bin + 1);
        let new_bin_location_2 : isize = new_bin_location_1 + this_size;

        // We give the memory in bin1 to the user immediately, but we save bin2 for ourselves
        let new_node : AllocatorNode& = &__builtin_int_to_ptr<AllocatorNode&>(new_bin_location_2);
        new_node.integrity_check = integrity_check;
        new_node.next_node = 0;
        allocator.bins[bin] = make_optional<Ptr<AllocatorNode>>({&new_node});

        return new_bin_location_1;
    }

    // FIXME: we generate invalid code without this return
    return 0;
};

function initialize_allocator : () -> Allocator = {
    let allocator : Allocator;

    // TODO: use variable initialized inside for loop
    let i : int = 0;

    // TODO: use i < 9
    // TODO: use i < allocator:len() (fails since it cannot dedude arg types)
    while (9 >= i) {
        // TODO: use a true random number
        // TODO: allow 64 bit constants
        allocator.integrity_checks[i] = 0xd2f4bd5b;


        // TODO: use constructor
        allocator.bins[i] = make_optional<Ptr<AllocatorNode>>();

        // TODO: i += 1
        i = i + 1;
    }
    return allocator;
};




function[T] sizeof : () -> isize = {
    return __builtin_sizeof<T>();
};

function[T] alignof : () -> isize = {
    return __builtin_alignof<T>();
};

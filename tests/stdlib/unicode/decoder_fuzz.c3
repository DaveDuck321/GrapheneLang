@require_once "std/unicode.c3"
@require_once "std/arithmetic.c3"
@require_once "std/io.c3"
@require_once "std/iterators.c3"
@require_once "std/span.c3"
@require_once "std/assert.c3"
@require_once "std/memory.c3"
@require_once "std/vector.c3"

function is_in_unicode_range : (scalar: UnicodeScalar) -> bool = {
    return is_in_interval(scalar.value, 0x0000, 0xD7FF)
            or is_in_interval(scalar.value, 0xE000, 0xFFFF)
            or is_in_interval(scalar.value, 0x10000, 0x10FFFF);
}

function test_for_correct_replacement_substitution : (allocator: Allocator mut&, data_view: View<u8>) -> void = {
    // Note: no attempt is made to verify the maximal subpart substitution algorithm is actually maximal.
    // However, this test does demonstrate that the substitutions are standard compliant.
    mut full_seq: Vector<UnicodeScalar> = make<Vector<UnicodeScalar>>(&mut allocator);
    full_seq:extend_back(make<UTF8DecodeIter>(data_view));
    runtime_assert(full_seq:length() <= 4);

    mut trunc_seq: Vector<UnicodeScalar> = make<Vector<UnicodeScalar>>(&mut allocator);
    trunc_seq:extend_back(make<UTF8DecodeIter>(data_view:slice_from(1)));
    runtime_assert(full_seq:length() <= 4);

    {
        // Test that all mappings are unique and fall within the valid unicode range
        runtime_assert(is_in_unicode_range(full_seq:get(0)));

        if full_seq:get(0) != invalid_unicode_scalar() {
            mut buffer: u8[4];
            let encoded: View<u8> = encode_to_utf8(full_seq:get(0), &mut buffer);
            runtime_assert(encoded == data_view:slice_to(encoded:length()));
        }
    }

    {
        // A valid UTF-8 sequence at any offset in the text should be decoded correctly
        // regardless of any surrounding substitutions.

        mut current_index_in_trunc_seq : isize = 0;

        // Ignore the first decoded character since we just deleted/ corrupted it
        // TODO: this should have automatic constant lifetime extension
        let full_seq_view: View<UnicodeScalar> = full_seq:view();
        let full_seq_view_tail: View<UnicodeScalar> = full_seq_view:slice_from(1);
        for full_scalar in full_seq_view_tail:get_iter() {
            if full_scalar == invalid_unicode_scalar() {
                continue;
            }

            mut found_matching_scalar: bool = false;
            while (current_index_in_trunc_seq < trunc_seq:length()) {
                let trunc_scalar : UnicodeScalar = trunc_seq:get(current_index_in_trunc_seq);
                current_index_in_trunc_seq += 1;

                if trunc_scalar != invalid_unicode_scalar() {
                    runtime_assert(trunc_scalar == full_scalar);
                    found_matching_scalar = true;
                    break;
                }
            }
            runtime_assert(found_matching_scalar);
        }

        // Make sure everything else is identified as an invalid symbol
        for i in range(current_index_in_trunc_seq, trunc_seq:length()) {
            runtime_assert(trunc_seq:get(i) == invalid_unicode_scalar());
        }
    }
}

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();

    // Fuzz the unicode parser with random bytes to verify incorrect codepoints
    // are correctly rejected. In -O3, an exhaustive test is on the cusp of being
    // viable in a reasonable time... Unfortunately the LLVM JIT makes this
    // infeasible even if we were to test with a reduced state-space.

    // Note: we don't have a seed to log.
    // If this ever fails please open an issue!!!!!
    let dev_random: Optional<File> = open({"/dev/random\0"}, O_RDONLY());
    runtime_assert(dev_random:has_value());

    let buffer: Span<u8> = allocator:allocate_span<u8>(1000000);
    let random_bytes: View<u8> = read(&dev_random:data(), buffer);

    for offset in range(random_bytes:length() - 4) {
        test_for_correct_replacement_substitution(&mut allocator, random_bytes:slice_between(offset, offset + 4));
    }

    allocator:deallocate_span(buffer);
    return 0;
}

/// @COMPILE
/// @RUN

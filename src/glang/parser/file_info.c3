@require_once "std/arithmetic.c3"
@require_once "std/algorithms.c3"
@require_once "std/memory.c3"
@require_once "std/string.c3"
@require_once "std/vector.c3"

typedef PositionLookupTable : {
    offsets : Vector<isize>
}

typedef FilePosition : {
    line : isize,
    column : isize,
    offset : isize,
}

function generate_position_lookup_table : (program : StringView, allocator: Allocator&) -> PositionLookupTable = {
    let new_lines : Vector<isize> = make_vector<isize>(&allocator);

    for current_offset in range(program.length) {
        if program.data[current_offset] == "\n"[0] {
            new_lines:push_back(current_offset);
        }
    }
    return {new_lines};
}

function lookup_zero_indexed_line_number : (table : PositionLookupTable&, offset : isize) -> isize = {
    return table.offsets:data():bisect_left(offset);
}

function lookup_position : (table: PositionLookupTable&, offset : isize) -> FilePosition = {
    let line_number : isize = table:lookup_zero_indexed_line_number(offset);

    // TODO: I'm using this pattern a lot, it should be a single statement
    let column_number : isize;
    if line_number == 0 {
        column_number = offset + 1;
    } else {
        column_number = offset - table.offsets:get(line_number - 1);
    }
    return {line_number + 1, column_number, offset};
}

@require_once "std/arithmetic.c3"
@require_once "std/memory.c3"
@require_once "std/string.c3"
@require_once "std/vector.c3"

typedef PositionLookupTable : {
    offsets : Vector<isize>
}

typedef FilePosition : {
    line_number : isize,
    column_number : isize,
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

function lookup_line_number : (table : PositionLookupTable&, offset : isize) -> isize = {
    // TODO: this can be a binary search
    for line in range(table.offsets.length - 1) {
        if table.offsets:get(line + 1) > offset {
            return line;
        }
    }
    return table.offsets.length - 1;
}

function lookup_position : (table: PositionLookupTable&, offset : isize) -> FilePosition = {
    let line_number : isize = table:lookup_line_number(offset);
    let column_number : isize = offset - table.offsets:get(line_number);
    return {line_number, column_number, offset};
}

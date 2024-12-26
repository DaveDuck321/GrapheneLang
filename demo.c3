@require_once "std/process.c3"
@require_once "std/format.c3"
@require_once "std/memory.c3"
@require_once "std/string.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();
    let buffer : Span<u8> = allocator:allocate_span<u8>(1024);

    let args : CString[1] = { make<CString>("/\0") };
    let process : Optional<Process> = execute({ "/usr/bin/ls\0" }, make<View<CString>>(&args, 1));

    if !process:has_value() {
        print(sv("Failed to start process"));
        return 1;
    }

    printf(sv("Process pid %"), process:data().pid);

    process:data().stdout:read(buffer);

    printf(sv("Process stdout '%'"), make<StringView>(buffer:to_view()));

    allocator:deallocate_span(buffer);

    return 0;
}

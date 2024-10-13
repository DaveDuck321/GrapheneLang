@require_once "std/fcntl.c3"
@require_once "std/format.c3"
@require_once "std/io.c3"
@require_once "std/json.c3"
@require_once "std/memory.c3"
@require_once "std/process.c3"
@require_once "std/sys/socket.c3"

@require_once "ring_buffer.c3"

typedef State : {
    allocator : Allocator,
    // Buffers for IPC (one per process).
    client_buffer : RingBuffer,
    parser_buffer : RingBuffer,
    parser_path : CStringView,
}

function make<State> : (parser_path : CStringView) -> Optional<State> = {
    let client_buffer : Optional<RingBuffer> =
        make<RingBuffer>(10 * 1024 * 1024, make<CString>("client_buffer\0"));
    if !client_buffer:has_value() {
        return make<Optional<State>>();
    }

    let parser_buffer : Optional<RingBuffer> =
        make<RingBuffer>(10 * 1024 * 1024, make<CString>("parser_buffer\0"));
    if !parser_buffer:has_value() {
        // FIXME deallocate the client buffer.
        return make<Optional<State>>();
    }

    return make<Optional<State>>({
        initialize_allocator(),
        client_buffer:data(),
        parser_buffer:data(),
        parser_path,
    });
}

typedef RPCRequest : {
    id : Optional<StringView>, // TODO parse to int.
    method : StringView, // TODO string.
    params : JSON_Node, // TODO mut&?
}

typedef RPCNotification : {
    notification : StringView, // TODO string.
    params       : JSON_Node, // TODO mut&?
}

function [IO] write : (io : IO mut&, req : RPCRequest) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    io:writef(sv("RPCRequest {id=%, method=%, params=%}"), req.id, req.method, req.params);
}

typedef StringStream : {
    buffer : u8[4096],
    buffer_pointer : isize,
}

function write : (ss : StringStream mut&, sv : StringView) -> void = {
    // TODO bounds check.
    for i in range(sv:length()) {
        ss.buffer[ss.buffer_pointer + i] = sv.buffer.data[i];
    }

    ss.buffer_pointer += sv:length();
}

function write : (ss : StringStream mut&, number : isize) -> void = {
    if number == 0 {
        ss.buffer[ss.buffer_pointer] = "0"[0];
        ss.buffer_pointer += 1;

        return;
    }

    let is_negative : bool = number < 0;
    mut current_number : isize = number;
    mut length : isize = 0;
    while current_number != 0 {
        let next_digit: u8 = as_logical(__builtin_narrow<i8>(abs(current_number % 10) + as_arithmetic("0"[0])));

        ss.buffer[ss.buffer_pointer + length] = next_digit;

        current_number /= 10;
        length += 1;
    }

    if is_negative {
        ss.buffer[length] = "-"[0];
        length += 1;
    }

    let span : Span<u8> = make<Span<u8>>(&mut ss.buffer, 4096);
    let result : Span<u8> = span:slice_between(ss.buffer_pointer, ss.buffer_pointer + length);
    result:reverse();
    ss.buffer_pointer += length;
}

function length : (ss : StringStream&) -> isize = {
    return ss.buffer_pointer;
}

function data : (ss : StringStream&) -> u8[&] = {
    return &ss.buffer;
}

function as_sv : (ss : StringStream&) -> StringView = {
    return make<StringView>(make<View<u8>>(&ss:data(), ss:length()));
}

function parse_natural_number : (cursor : Cursor mut&) -> Optional<int> = {
    mut number : int = 0;
    mut valid : bool = false;

    while cursor:has_next() {
        // TODO this has to be placed on the stack explicitly. Is this
        // desirable?
        let digit : Optional<u8> = cursor:peek();
        if !ascii_is_digit(digit:data()) {
            break;
        }

        cursor:next();

        number *= 10;
        number += as_arithmetic(digit:data()) - as_arithmetic(char("0"));
        valid = true;
    }

    if valid {
        return make<Optional<int>>(number);
    }

    return make<Optional<int>>();
}

function handle_next_rpc : (
    state : State mut&, in_file : File, out_file : File
) -> bool = {
    mut view_opt : Optional<View<u8>> = state.client_buffer:read_line(in_file);
    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        return false;
    }

    mut view : StringView = make<StringView>(view_opt:data());
    if view:length() == 0 {
        print(sv("Received EOF"));
        return false;
    }

    // TODO stop using stuff from json.c3 (move to e.g. parse.c3?).
    mut cursor : Cursor = { view:get_iter(), &mut state.allocator };

    if !cursor:eat(sv("Content-Length: ")) {
        print(sv("Message does not start with 'Content-Length: '"));
        return false;
    }

    // TODO use this to buffer RPCs correctly.
    let length : Optional<int> = cursor:parse_natural_number();

    if !length:has_value() {
        print(sv("Failed to parse length"));
        return false;
    }

    // 2 newlines, and then JSON.
    if !cursor:eat(sv("\r\n")) {
        print(sv("Failed to eat newline"));
        return false;
    }

    state.client_buffer:advance(view:length());
    view_opt = state.client_buffer:read_line(in_file);
    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        return false;
    }

    view = make<StringView>(view_opt:data());
    if view:length() == 0 {
        print(sv("Received EOF"));
        return false;
    }

    if view != sv("\r\n") {
        print(sv("Failed to eat second newline"));
        return false;
    }

    state.client_buffer:advance(view:length());
    view_opt = state.client_buffer:read(in_file, length:data());
    if !view_opt:has_value() {
        print(sv("Failed to read length"));
        return false;
    }

    view = { .buffer = view_opt:data() };
    mut json : Optional<JSON_Node> = json_parse(view, &mut state.allocator);
    if !json:has_value() {
        print(sv("Failed to parse json"));
        return false;
    }

    state.client_buffer:advance(view:length());

    // TODO this is all kinda horrible.
    let id : Optional<JSON_Node> = json:data().map:at(sv("id"));
    let method : Optional<JSON_Node> = json:data().map:at(sv("method"));
    mut params : Optional<JSON_Node> = json:data().map:at(sv("params"));

    mut deallocate_params : bool = false;
    if !params:has_value() {
        // TODO Should the RPC objects take an optional instead?
        params = make<Optional<JSON_Node>>(make<JSON_Node>(/* object */ 6, &mut state.allocator));
        deallocate_params = true;
    }

    if id:has_value() {
        let request : RPCRequest = {
            .id = id:data().value,
            .method = method:data().value:data(),
            .params = params:data(),
        };
        state:dispatch_rpc_request(request, out_file);
    }
    else {
        let notification : RPCNotification = {
            .notification = method:data().value:data(),
            .params = params:data(),
        };
        state:dispatch_rpc_notification(notification, out_file);
    }

    json:data():deallocate();
    if deallocate_params {
        params:data():deallocate();
    }

    return true;
}

function handle_initialize : (state : State mut&, req : RPCRequest, out : BufferedFile mut&) -> void = {
    printf(sv("Received initialize request with params=%"), req.params);

    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"capabilities\":{\"diagnosticProvider\":{\"interFileDependencies\":true,\"workspaceDiagnostics\":false}, \"textDocumentSync\": {\"change\": 1}}}}");
    out:writef(sv("Content-Length: %\r\n"), resp:length());
    out:write(sv("\r\n"));
    out:write(resp);
    out:flush();
}

function handle_shutdown : (state : State mut&, req : RPCRequest, out : BufferedFile mut&) -> void = {
    print(sv("Received shutdown request"));

    // FIXME I think this either asserts or exit doesn't exit correctly. Need to
    // debug with gdb/strace.
    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%}");
    mut ss : StringStream;
    ss.buffer_pointer = 0;
    ss:writef(resp, req.id:data());
    out:writef(sv("Content-Length: %\r\n"), ss:length());
    out:write(sv("\r\n"));
    out:write(ss:as_sv());
    out:flush();
}

function handle_exit : (state : State mut&, req : RPCNotification, out : BufferedFile mut&) -> void = {
    print(sv("Received exit notification"));

    // TODO exit with 1 if shutdown wasn't received.
    sys_exit(0);
}

function handle_diagnostic : (
    state : State mut&, req : RPCRequest, out : BufferedFile mut&
) -> void = {
    printf(sv("Received diagnostic request with params=%"), req.params);

    let text_document : Optional<JSON_Node> = req.params.map:at(sv("textDocument"));
    if !text_document:has_value() {
        print(sv("Bad diagnostic request 1"));
        return;
    }

    let uri : Optional<JSON_Node> = text_document:data().map:at(sv("uri"));
    if !uri:has_value() {
        print(sv("Bad diagnostic request 2"));
        return;
    }
    if !uri:data().value:has_value() {
        print(sv("Bad diagnostic request 3"));
        return;
    }

    // We expect `file:///file/system/path`.
    let protocol : StringView = uri:data().value:data():slice_to(7);
    if protocol != sv("file://") {
        printf(sv("Unsupported URI protocol `%`"), protocol);
        return;
    }

    let path : StringView = uri:data().value:data():slice_from(7);
    printf(sv("Reading from `%`"), path);

    mut path_ss : StringStream;
    path_ss.buffer_pointer = 0;
    path_ss:writef(sv("%\0"), path);

    let args : CString[1] = { { &path_ss:data() } };

    let process : Optional<Process> = execute(
        { &state.parser_path:data() },
        make<View<CString>>(&args)
    );
    if !process:has_value() {
        print(sv("Failed to start process"));
        return;
    }

    // This buffer is reused across multiple, independent calls to the parser.
    // We need to make sure that the output of one invocation doesn't affect
    // subsequent invocations.
    state.parser_buffer:clear();

    mut view_opt : Optional<View<u8>> = state.parser_buffer:read_line(process:data().stdout);
    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        return;
    }
    let view : StringView = make<StringView>(view_opt:data());

    mut json : Optional<JSON_Node> = json_parse(view, &mut state.allocator);
    if !json:has_value() {
        print(view);
        print(sv("Failed to parse json"));
        return;
    }

    let success : Optional<JSON_Node> = json:data().map:at(sv("success"));
    if !success:has_value() {
        // The parser didn't output an error, so the code is good. We still need
        // to reply though.
        let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{\"kind\":\"full\", \"items\": []}}");
        mut ss : StringStream;
        ss.buffer_pointer = 0;
        ss:writef(resp, req.id:data());
        print(ss:as_sv());
        out:writef(sv("Content-Length: %\r\n"), ss:length());
        out:write(sv("\r\n"));
        out:write(ss:as_sv());
        out:flush();
        json:data():deallocate();
        return;
    }

    // TODO need an actual API here.
    if success:data().type != /* false */ 2 {
        json:data():deallocate();
        return;
    }

    let message : Optional<JSON_Node> = json:data().map:at(sv("message"));
    if !message:has_value() {
        print(sv("'message' not found"));
        json:data():deallocate();
        return;
    }

    let location : Optional<JSON_Node> = json:data().map:at(sv("location"));
    if !location:has_value() {
        print(sv("'location' not found"));
        json:data():deallocate();
        return;
    }

    let line : Optional<JSON_Node> = location:data().map:at(sv("line"));
    if !line:has_value() {
        print(sv("'line' not found"));
        json:data():deallocate();
        return;
    }

    let column : Optional<JSON_Node> = location:data().map:at(sv("column"));
    if !column:has_value() {
        print(sv("'column' not found"));
        json:data():deallocate();
        return;
    }

    mut cursor : Cursor = { line:data().value:data():get_iter(), &mut state.allocator };
    let line_number : Optional<int> = cursor:parse_natural_number();

    if !line_number:has_value() {
        print(sv("Couldn't parse line number"));
        json:data():deallocate();
        return;
    }

    cursor.iter = column:data().value:data():get_iter();
    let column_number : Optional<int> = cursor:parse_natural_number();

    if !column_number:has_value() {
        print(sv("Couldn't parse column number"));
        json:data():deallocate();
        return;
    }

    printf(sv("message: %, line: %"), message:data().value:data(), line:data().value:data());
    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{\"kind\":\"full\", \"items\": [{ \"range\": { \"start\": { \"line\": %, \"character\": % }, \"end\" : { \"line\": %, \"character\": % } }, \"message\": \"%\" }]}}");
    mut ss : StringStream;
    ss.buffer_pointer = 0;
    ss:writef(resp, req.id:data(), line_number:data() - 1, column_number:data() - 1, line_number:data() - 1, column_number:data() - 1, message:data().value:data());
    print(ss:as_sv());
    out:writef(sv("Content-Length: %\r\n"), ss:length());
    out:write(sv("\r\n"));
    out:write(ss:as_sv());
    out:flush();

    json:data():deallocate();
}

function handle_text_change : (
    state : State mut&, req : RPCNotification, out : BufferedFile mut&
) -> void = {
    printf(sv("Received textDocument/didChange notification"));

    let text_document : Optional<JSON_Node> = req.params.map:at(sv("textDocument"));
    if !text_document:has_value() {
        print(sv("textDocument not found"));
        return;
    }

    let uri : Optional<JSON_Node> = text_document:data().map:at(sv("uri"));
    if !uri:has_value() {
        print(sv("uri not found"));
        return;
    }

    let version : Optional<JSON_Node> = text_document:data().map:at(sv("version"));
    if !version:has_value() {
        print(sv("version not found"));
        return;
    }

    let content_changes : Optional<JSON_Node> = req.params.map:at(sv("contentChanges"));
    if !content_changes:has_value() {
        print(sv("contentChanges not found"));
        return;
    }

    if content_changes:data().children:length() == 0 {
        print(sv("contentChanges is empty"));
        return;
    }

    let text_wrapper : JSON_Node = content_changes:data().children:get(0);
    let text : Optional<JSON_Node> = text_wrapper.map:at(sv("text"));
    if !text:has_value() {
        print(sv("text not found"));
        return;
    }

    // Share the new file contents via an anonymous file.
    let fd : int = memfd_create(make<CString>("shared_file\0"), 0x0);
    if fd < 0 {
        print(sv("memfd_create failed"));
        return;
    }

    let text_length : isize = text:data().value:data():length();

    if ftruncate(fd, text_length) != 0 {
        print(sv("ftruncate failed"));
        return;
    }

    // TODO we could seal the file before passing it to a different process.
    let addr : iptr = mmap(
        0, text_length, SYS_PROT_READ() | SYS_PROT_WRITE(),
        SYS_MAP_SHARED(), fd, 0
    );
    let text_buffer : u8[mut&] = &mut addr_to_mut_heap_array<u8>(addr);
    memcpy(&mut text_buffer, &text:data().value:data():data(), text_length);

    mut path : StringStream;
    path.buffer_pointer = 0;
    path:writef(sv("/proc/%/fd/%\0"), getpid(), fd);
    printf(sv("Shared fd path: %"), path:as_sv());

    let args : CString[1] = { { &path:data() } };

    let process : Optional<Process> = execute(
        { &state.parser_path:data() },
        make<View<CString>>(&args)
    );
    if !process:has_value() {
        print(sv("Failed to start process"));
        munmap(addr, text_length);
        sys_close(fd);
        return;
    }

    // This buffer is reused across multiple, independent calls to the parser.
    // We need to make sure that the output of one invocation doesn't affect
    // subsequent invocations.
    state.parser_buffer:clear();

    mut view_opt : Optional<View<u8>> = state.parser_buffer:read_line(process:data().stdout);
    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        munmap(addr, text_length);
        sys_close(fd);
        return;
    }
    let view : StringView = make<StringView>(view_opt:data());

    mut json : Optional<JSON_Node> = json_parse(view, &mut state.allocator);
    if !json:has_value() {
        print(sv("Failed to parse json"));
        munmap(addr, text_length);
        sys_close(fd);
        return;
    }

    munmap(addr, text_length);
    sys_close(fd);

    let success : Optional<JSON_Node> = json:data().map:at(sv("success"));
    if !success:has_value() {
        // The parser didn't output an error, so the code is good. We still need
        // to reply though.
        let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{\"uri\": \"%\", \"version\": %, \"diagnostics\": []}}");
        mut ss : StringStream;
        ss.buffer_pointer = 0;
        ss:writef(resp, uri:data().value:data(), version:data().value:data());
        print(ss:as_sv());
        out:writef(sv("Content-Length: %\r\n"), ss:length());
        out:write(sv("\r\n"));
        out:write(ss:as_sv());
        out:flush();
        json:data():deallocate();
        return;
    }

    // TODO need an actual API here.
    if success:data().type != /* false */ 2 {
        json:data():deallocate();
        return;
    }

    let message : Optional<JSON_Node> = json:data().map:at(sv("message"));
    if !message:has_value() {
        print(sv("'message' not found"));
        json:data():deallocate();
        return;
    }

    let location : Optional<JSON_Node> = json:data().map:at(sv("location"));
    if !location:has_value() {
        print(sv("'location' not found"));
        json:data():deallocate();
        return;
    }

    let line : Optional<JSON_Node> = location:data().map:at(sv("line"));
    if !line:has_value() {
        print(sv("'line' not found"));
        json:data():deallocate();
        return;
    }

    let column : Optional<JSON_Node> = location:data().map:at(sv("column"));
    if !column:has_value() {
        print(sv("'column' not found"));
        json:data():deallocate();
        return;
    }

    mut cursor : Cursor = { line:data().value:data():get_iter(), &mut state.allocator };
    let line_number : Optional<int> = cursor:parse_natural_number();

    if !line_number:has_value() {
        print(sv("Couldn't parse line number"));
        json:data():deallocate();
        return;
    }

    cursor.iter = column:data().value:data():get_iter();
    let column_number : Optional<int> = cursor:parse_natural_number();

    if !column_number:has_value() {
        print(sv("Couldn't parse column number"));
        json:data():deallocate();
        return;
    }

    printf(sv("message: %, line: %"), message:data().value:data(), line:data().value:data());
    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{\"uri\": \"%\", \"version\": %, \"diagnostics\": [{ \"range\": { \"start\": { \"line\": %, \"character\": % }, \"end\" : { \"line\": %, \"character\": % } }, \"message\": \"%\" }]}}");
    mut ss : StringStream;
    ss.buffer_pointer = 0;
    ss:writef(resp, uri:data().value:data(), version:data().value:data(), line_number:data() - 1, column_number:data() - 1, line_number:data() - 1, column_number:data() - 1, message:data().value:data());
    print(ss:as_sv());
    out:writef(sv("Content-Length: %\r\n"), ss:length());
    out:write(sv("\r\n"));
    out:write(ss:as_sv());
    out:flush();

    json:data():deallocate();
}

function dispatch_rpc_request : (
    state : State mut&, req : RPCRequest, fd : File
) -> void = {
    // FIXME we need a better way; mostly because our formatters don't work with
    // raw `File`s.
    mut out : BufferedFile;
    out.underlying = fd;
    out.buffer_pointer = 0;

    // TODO this should work without the sv().
    if req.method == sv("initialize") {
        state:handle_initialize(req, &mut out);
        return;
    }

    if req.method == sv("textDocument/diagnostic") {
        state:handle_diagnostic(req, &mut out);
        return;
    }

    if req.method == sv("shutdown") {
        state:handle_shutdown(req, &mut out);
    }

    printf(sv("Unsupported method: %"), req.method);
}

function dispatch_rpc_notification : (
    state : State mut&, req : RPCNotification, fd : File
) -> void = {
    // FIXME we need a better way; mostly because our formatters don't work with
    // raw `File`s.
    mut out : BufferedFile;
    out.underlying = fd;
    out.buffer_pointer = 0;

    if req.notification == sv("textDocument/didChange") {
        state:handle_text_change(req, &mut out);
        return;
    }

    if req.notification == sv("exit") {
        state:handle_exit(req, &mut out);
    }

    printf(sv("Unsupported notification: %"), req.notification);
}

function open_socket : (path : CStringView) -> Optional<File> = {
    // TODO sizeof function that takes struct members?
    if path:length() > 108 {
        return make<Optional<File>>();
    }

    let socket : Optional<File> = socket(AF_UNIX(), SOCK_STREAM(), 0);
    if !socket:has_value() {
        return make<Optional<File>>();
    }

    mut sockaddr : sockaddr_un;
    // TODO the type conversions here are horrible.
    sockaddr.sun_family = Narrow<sa_family_t>(as_logical(AF_UNIX()));
    memset(&mut sockaddr.sun_path, char("\0"), sockaddr.sun_path:length());
    memcpy(&mut sockaddr.sun_path, &path:data(), path:length());

    let ret : int = connect(socket:data(), &sockaddr);
    if ret < 0 {
        socket:data():close();
        return make<Optional<File>>();
    }

    return socket;
}

function main : (argc : int, argv : u8[&][&]) -> int = {
    // Default to stdio if no arguments are passed in.
    mut in_file : File = { .fd = /* stdin */ 0 };
    mut out_file : File = { .fd = /* stderr */ 2 };

    mut parser_path : CStringView;

    mut saw_pipe_arg : bool = false;
    mut saw_parser_arg : bool = false;

    // If --pipe was passed, use a Unix socket instead.
    for i in range(1, argc) {
        let arg : CStringView = make<CStringView>(&argv[i]);

        if arg:slice_to(length("--pipe=")) == sv("--pipe=") {
            if saw_pipe_arg {
                printf(sv("Error: --pipe= can only appear once"));
                return 1;
            }
            saw_pipe_arg = true;

            let sock_path : CStringView = arg:slice_from(length("--pipe="));
            let sock_file : Optional<File> = open_socket(sock_path);
            if !sock_file:has_value() {
                printf(sv("Error: could not open socket from %"), sock_path);
                return 1;
            }
            in_file = sock_file:data();
            out_file = sock_file:data();
        }
        else {
            if arg:slice_to(length("--parser=")) == sv("--parser=") {
                if saw_parser_arg {
                    printf(sv("Error: --parser= can only appear once"));
                    return 1;
                }
                saw_parser_arg = true;

                parser_path = arg:slice_from(length("--parser="));
            }
            else{
                printf(sv("Error: unknown argument '%'"), arg);
                return 1;
            }
        }
    }

    if !(saw_pipe_arg and saw_parser_arg) {
        if argc >= 1 {
            printf(
                sv("Usage: % --pipe=/path/to/socket --parser=/path/to/parser"),
                make<CStringView>(&argv[0])
            );
        }
        return 1;
    }

    print(sv("Starting!"));

    mut state : Optional<State> = make<State>(parser_path);
    if !state:has_value() {
        print(sv("Error: failed to initialize state"));
        return 1;
    }

    while true {
        if !state:data():handle_next_rpc(in_file, out_file) {
            // Try and resync if something went wrong.
            state:data().client_buffer:clear();
        }
    }

    return 0;
}

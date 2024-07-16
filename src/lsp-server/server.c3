@require_once "std/fcntl.c3"
@require_once "std/format.c3"
@require_once "std/io.c3"
@require_once "std/json.c3"
@require_once "std/memory.c3"
@require_once "std/process.c3"
@require_once "std/sys/socket.c3"

@require_once "ring_buffer.c3"

typedef State : {
    // Buffers for IPC (one per process).
    client_buffer : RingBuffer,
    parser_buffer : RingBuffer,
}

function make<State> : () -> Optional<State> = {
    let client_buffer : Optional<RingBuffer> =
        make<RingBuffer>(262144, make<CString>("client_buffer\0"));
    if !client_buffer:has_value() {
        return make<Optional<State>>();
    }

    let parser_buffer : Optional<RingBuffer> =
        make<RingBuffer>(262144, make<CString>("parser_buffer\0"));
    if !parser_buffer:has_value() {
        // FIXME deallocate the client buffer.
        return make<Optional<State>>();
    }

    return make<Optional<State>>({client_buffer:data(), parser_buffer:data()});
}

typedef RPCRequest : {
    id : StringView, // TODO parse to int.
    method : StringView, // TODO string.
    params : JSON_Node, // TODO mut&?
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

// FIXME we really do want to take this by reference.
function [IO] write : (io : IO mut&, ss : StringStream) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    let span : View<u8> = make<View<u8>>(&ss.buffer, ss.buffer_pointer);
    io:write(make<StringView>(span));
}

function parse_natural_number : (cursor : Cursor mut&) -> Optional<int> = {
    mut number : int = 0;
    mut valid : bool = false;

    while cursor:has_next() {
        // TODO this has to be placed on the stack explicitly. Is this a good
        // thing or a bad thing?
        let digit : Optional<u8> = cursor:peak();
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

function parse_next_rpc : (
    state : State mut&, fd : File, allocator : Allocator mut&
) -> Optional<RPCRequest> = {
    mut view_opt : Optional<View<u8>> = state.client_buffer:read_line(fd);
    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        return make<Optional<RPCRequest>>();
    }

    mut view : StringView = { .buffer = view_opt:data() };
    if view:length() == 0 {
        print(sv("Received EOF"));
        return make<Optional<RPCRequest>>();
    }

    // TODO stop using stuff from json.c3 (move to e.g. parse.c3?).
    mut cursor : Cursor = { view:get_iter(), &mut allocator };

    if !cursor:eat(sv("Content-Length: ")) {
        print(sv("Message does not start with 'Content-Length: '"));
        return make<Optional<RPCRequest>>();
    }

    // TODO use this to buffer RPCs correctly.
    let length : Optional<int> = cursor:parse_natural_number();

    if !length:has_value() {
        print(sv("Failed to parse length"));
        return make<Optional<RPCRequest>>();
    }

    printf(sv("Content length=%"), length:data());

    // 2 newlines, and then JSON.
    if !cursor:eat(sv("\r\n")) {
        print(sv("Failed to eat newline"));
        return make<Optional<RPCRequest>>();
    }

    state.client_buffer:advance(view:length());
    view_opt = state.client_buffer:read_line(fd);
    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        return make<Optional<RPCRequest>>();
    }

    view = { .buffer = view_opt:data() };
    if view:length() == 0 {
        print(sv("Received EOF"));
        return make<Optional<RPCRequest>>();
    }

    if view != sv("\r\n") {
        print(sv("Failed to eat second newline"));
        return make<Optional<RPCRequest>>();
    }

    state.client_buffer:advance(view:length());
    view_opt = state.client_buffer:read(fd, length:data());
    if !view_opt:has_value() {
        print(sv("Failed to read length"));
        return make<Optional<RPCRequest>>();
    }

    view = { .buffer = view_opt:data() };
    let json : Optional<JSON_Node> = json_parse(view, &mut allocator);
    if !json:has_value() {
        print(sv("Failed to parse json"));
        return make<Optional<RPCRequest>>();
    }

    state.client_buffer:advance(view:length());

    printf(sv("Received %"), json:data());

    // TODO this is all kinda horrible.
    let id : Optional<JSON_Node> = json:data().map:at(sv("id"));
    let method : Optional<JSON_Node> = json:data().map:at(sv("method"));
    let params : Optional<JSON_Node> = json:data().map:at(sv("params"));

    if !id:has_value() {
        print(sv("Ignoring RPC notification"));
        return make<Optional<RPCRequest>>();
    }

    return make<Optional<RPCRequest>>({
        .id = id:data().value:data(),
        .method = method:data().value:data(),
        .params = params:data(),
    });
}

function handle_initialize : (params : JSON_Node) -> void = {
    printf(sv("Received initialize request with params=%"), params);
}

function handle_diagnostic : (
    state : State mut&, req : RPCRequest, out : BufferedFile
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

    mut path_buffer : u8[512];
    for i in range(512) {
        path_buffer[i] = Narrow<u8>(0x00);
    }
    for i in range(path:length()) {
        path_buffer[i] = path.buffer.data[i];
    }

    let path_cstring : CString[1] = { make<CString>(&path_buffer) };

    let process : Optional<Process> = execute(
        make<CString>("/var/home/antros/Source/GrapheneLang/dist/parser\0"),
        make<View<CString>>(&path_cstring, 1)
    );
    if !process:has_value() {
        print(sv("Failed to start process"));
        return;
    }

    mut allocator : Allocator = initialize_allocator();

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

    let json : Optional<JSON_Node> = json_parse(view, &mut allocator);
    if !json:has_value() {
        print(sv("Failed to parse json"));
        return;
    }

    printf(sv("Received %"), json:data());
    let success : Optional<JSON_Node> = json:data().map:at(sv("success"));
    if !success:has_value() {
        // The parser didn't output an error, so the code is good. We still need
        // to reply though.
        let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{\"kind\":\"full\", \"items\": []}}");
        mut ss : StringStream;
        ss.buffer_pointer = 0;
        ss:writef(resp, req.id);
        print(ss);
        out:writef(sv("Content-Length: %\r\n"), ss:length());
        out:write(sv("\r\n"));
        out:write(ss);
        out:flush();
        return;
    }

    // TODO need an actual API here.
    if success:data().type != /* false */ 2 {
        return;
    }

    let message : Optional<JSON_Node> = json:data().map:at(sv("message"));
    if !message:has_value() {
        print(sv("'message' not found"));
        return;
    }

    let location : Optional<JSON_Node> = json:data().map:at(sv("location"));
    if !location:has_value() {
        print(sv("'location' not found"));
        return;
    }

    let line : Optional<JSON_Node> = location:data().map:at(sv("line"));
    if !line:has_value() {
        print(sv("'line' not found"));
        return;
    }

    let column : Optional<JSON_Node> = location:data().map:at(sv("column"));
    if !column:has_value() {
        print(sv("'column' not found"));
        return;
    }

    mut cursor : Cursor = { line:data().value:data():get_iter(), &mut allocator };
    let line_number : Optional<int> = cursor:parse_natural_number();

    if !line_number:has_value() {
        print(sv("Couldn't parse line number"));
        return;
    }

    cursor.iter = column:data().value:data():get_iter();
    let column_number : Optional<int> = cursor:parse_natural_number();

    if !column_number:has_value() {
        print(sv("Couldn't parse column number"));
        return;
    }

    printf(sv("message: %, line: %"), message:data().value:data(), line:data().value:data());
    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{\"kind\":\"full\", \"items\": [{ \"range\": { \"start\": { \"line\": %, \"character\": % }, \"end\" : { \"line\": %, \"character\": % } }, \"message\": \"%\" }]}}");
    mut ss : StringStream;
    ss.buffer_pointer = 0;
    ss:writef(resp, req.id, line_number:data() - 1, column_number:data() - 1, line_number:data() - 1, column_number:data() - 1, message:data().value:data());
    print(ss);
    out:writef(sv("Content-Length: %\r\n"), ss:length());
    out:write(sv("\r\n"));
    out:write(ss);
    out:flush();
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
        handle_initialize(req.params);

        let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"capabilities\":{\"diagnosticProvider\":{\"interFileDependencies\":true,\"workspaceDiagnostics\":false}}}}");
        out:writef(sv("Content-Length: %\r\n"), resp:length());
        out:write(sv("\r\n"));
        out:write(resp);
        out:flush();
        return;
    }

    if req.method == sv("textDocument/diagnostic") {
        state:handle_diagnostic(req, out);
        return;
    }

    printf(sv("Unsupported method: %"), req.method);
}

function open_socket : (path : CString) -> Optional<File> = {
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
    // TODO provide memset/memcpy.
    for i in range(path:length()) {
        sockaddr.sun_path[i] = path.data[i];
    }
    for i in range(path:length(), sockaddr.sun_path:length()) {
        sockaddr.sun_path[i] = Narrow<u8>(0x00);
    }

    let ret : int = connect(socket:data(), &sockaddr);
    if ret < 0 {
        socket:data():close();
        return make<Optional<File>>();
    }

    return socket;
}

function main : (argc : int, argv : u8[&][&]) -> int = {
    if argc > 2 {
        printf(sv("Call with --pipe=/path/to/sock"));
        return 1;
    }

    // Default to stdio if no arguments are passed in.
    mut in_file : File = { .fd = /* stdin */ 0 };
    mut out_file : File = { .fd = /* stderr */ 2 };

    // If --pipe was passed, use a Unix socket instead.
    if argc == 2 {
        // HACK need proper CString manipulation.
        let sock_path : CString = { &addr_to_heap_array<u8>(ref_to_addr(&argv[1]) + 7) };
        let sock_file : Optional<File> = open_socket(sock_path);
        if !sock_file:has_value() {
            let path : StringView = make<StringView>(sock_path);
            printf(sv("Error: could not open socket from %"), path);
            return 1;
        }
        in_file = sock_file:data();
        out_file = sock_file:data();
    }

    print(sv("Starting!"));

    mut allocator : Allocator = initialize_allocator();
    mut state : Optional<State> = make<State>();
    if !state:has_value() {
        print(sv("Error: failed to initialize state"));
        return 1;
    }

    // FIXME we're mmap'ing new ring buffers on every request...
    while true {
        let parsed : Optional<RPCRequest> = state:data():parse_next_rpc(in_file, &mut allocator);

        if parsed:has_value() {
            state:data():dispatch_rpc_request(parsed:data(), out_file);
        }
    }

    return 0;
}

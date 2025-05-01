@require_once "std/fcntl.c3"
@require_once "std/format.c3"
@require_once "std/io.c3"
@require_once "std/json.c3"
@require_once "std/memory.c3"
@require_once "std/process.c3"
@require_once "std/sstream.c3"
@require_once "std/sys/socket.c3"

@require_once "ring_buffer.c3"

typedef State : {
    allocator : Allocator,
    // Buffers for IPC (one per process).
    client_buffer : RingBuffer,
    parser_buffer : RingBuffer,
    parser_path : CStringView,
}

function make<State> : (parser_path : CStringView) -> State = {
    let client_buffer : RingBuffer =
        make<RingBuffer>(10 * 1024 * 1024, make<CString>("client_buffer\0"));

    let parser_buffer : RingBuffer =
        make<RingBuffer>(10 * 1024 * 1024, make<CString>("parser_buffer\0"));

    return {
        initialize_allocator(),
        client_buffer,
        parser_buffer,
        parser_path,
    };
}

typedef RPCRequest : {
    id : Optional<StringView>, // TODO parse to int.
    method : StringView, // TODO string.
    params : Optional<JSON_Node>, // TODO mut&?
}

typedef RPCNotification : {
    notification : StringView, // TODO string.
    params       : Optional<JSON_Node>, // TODO mut&?
}

typedef Diagnostic : {
    message : JSON_Node,
    line    : int,
    column  : int,
}

typedef ParserOutput : {
    diagnostic : Optional<Diagnostic>,
}

function [IO] write : (io : IO mut&, req : RPCRequest) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    io:writef(sv("RPCRequest {id=%, method=%, params=%}"), req.id, req.method, req.params);
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

function run_parser : (state : State mut&, path : StringView) -> Optional<JSON_Node> = {
    printf(sv("Running parser % %"), state.parser_path, path);

    // Append a null terminator to the path.
    mut arg : StringStream = make<StringStream>(&mut state.allocator);
    arg:writef(sv("%\0"), path);

    // The arugments are naturally an array, even if we only have one.
    let arg_sv : StringView = arg:as_sv();
    let args : CString[1] = { { &arg_sv:data() }};

    mut process : Optional<Process> = execute(
        { &state.parser_path:data() },
        make<View<CString>>(&args),
    );

    arg:deallocate();

    if !process:has_value() {
        print(sv("Failed to start process"));
        return make<Optional<JSON_Node>>();
    }

    // This buffer is reused across multiple, independent calls to the parser.
    // We need to make sure that the output of one invocation doesn't affect
    // subsequent invocations.
    state.parser_buffer:clear();

    mut view_opt : Optional<View<u8>> = state.parser_buffer:read_eof(process:data().stdout);

    let wstatus : Optional<int> = process:data():wait();
    if !wstatus:has_value() {
        print(sv("Failed to wait on process"));
    }
    else {
        if !WIFEXITED(wstatus:data()) {
            print(sv("Abnormal parser termination"));
        }
    }

    process:data():close();

    if !view_opt:has_value() {
        print(sv("Failed to read line"));
        return make<Optional<JSON_Node>>();
    }

    let view : StringView = make<StringView>(view_opt:data());

    mut json : Optional<JSON_Node> = json_parse(view, &mut state.allocator);
    if !json:has_value() {
        print(sv("Failed to parse json"));
        return make<Optional<JSON_Node>>();
    }

    return json;
}

function parse_parser_output : (
    state : State mut&, json : JSON_Node mut&
) -> Optional<ParserOutput> = {
    mut output : ParserOutput = { make<Optional<Diagnostic>>() };

    let success : Optional<JSON_Node> = json.map:at(sv("success"));

    // The parser didn't produce an error.
    if !success:has_value() {
        return make<Optional<ParserOutput>>(output);
    }

    // TODO need an actual API here.
    if success:data().type != /* false */ 2 {
        print(sv("Unexpected 'success' value"));
        return make<Optional<ParserOutput>>();
    }

    // Now we need to extract all diagnostic information.
    let message : Optional<JSON_Node> = json.map:at(sv("message"));
    if !message:has_value() {
        print(sv("'message' not found"));
        return make<Optional<ParserOutput>>();
    }

    let location : Optional<JSON_Node> = json.map:at(sv("location"));
    if !location:has_value() {
        print(sv("'location' not found"));
        return make<Optional<ParserOutput>>();
    }

    let line : Optional<JSON_Node> = location:data().map:at(sv("line"));
    if !line:has_value() {
        print(sv("'line' not found"));
        return make<Optional<ParserOutput>>();
    }

    let column : Optional<JSON_Node> = location:data().map:at(sv("column"));
    if !column:has_value() {
        print(sv("'column' not found"));
        return make<Optional<ParserOutput>>();
    }

    mut cursor : Cursor = { line:data().value:data():get_iter(), &mut state.allocator };
    let line_number : Optional<int> = cursor:parse_natural_number();

    if !line_number:has_value() {
        print(sv("Couldn't parse line number"));
        return make<Optional<ParserOutput>>();
    }

    cursor.iter = column:data().value:data():get_iter();
    let column_number : Optional<int> = cursor:parse_natural_number();

    if !column_number:has_value() {
        print(sv("Couldn't parse column number"));
        return make<Optional<ParserOutput>>();
    }

    let diagnostic : Diagnostic = {
        message:data(),
        line_number:data() - 1,
        column_number:data() - 1,
    };

    output.diagnostic = make<Optional<Diagnostic>>(diagnostic);

    return make<Optional<ParserOutput>>(output);
}

function send_response : (out : BufferedFile mut&, resp : StringView) -> void = {
    printf(sv("<-- %"), resp);

    // Write the Content-Length header required by the language server protocol
    // followed by a JSON-RPC message.
    out:writef(sv("Content-Length: %\r\n"), resp:length());
    out:write(sv("\r\n"));
    out:write(resp);
    out:flush();
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

    runtime_assert(!cursor:has_next());

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

    if id:has_value() {
        let request : RPCRequest = {
            .id = id:data().value,
            .method = method:data().value:data(),
            .params = params,
        };
        state:dispatch_rpc_request(request, out_file);
    }
    else {
        let notification : RPCNotification = {
            .notification = method:data().value:data(),
            .params = params,
        };
        state:dispatch_rpc_notification(notification, out_file);
    }

    json:data():deallocate();

    return true;
}

function handle_initialize : (state : State mut&, req : RPCRequest, out : BufferedFile mut&) -> void = {
    printf(sv("Received initialize request with params=%"), req.params);

    out:send_response(
        sv("{\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"capabilities\":{\"diagnosticProvider\":{\"interFileDependencies\":true,\"workspaceDiagnostics\":false}, \"textDocumentSync\": {\"change\": 1}}}}")
    );
}

function handle_shutdown : (state : State mut&, req : RPCRequest, out : BufferedFile mut&) -> void = {
    print(sv("Received shutdown request"));

    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{}}");
    mut ss : StringStream = make<StringStream>(&mut state.allocator);
    ss:writef(resp, req.id:data());
    out:send_response(ss:as_sv());
    ss:deallocate();
}

function handle_exit : (state : State mut&, req : RPCNotification, out : BufferedFile mut&) -> void = {
    print(sv("Received exit notification"));

    state.client_buffer:deallocate();
    state.parser_buffer:deallocate();

    // TODO exit with 1 if shutdown wasn't received.
    sys_exit(0);
}

function handle_diagnostic : (
    state : State mut&, req : RPCRequest, out : BufferedFile mut&
) -> void = {
    printf(sv("Received diagnostic request with params=%"), req.params);

    if !req.params:has_value() {
        print(sv("Missing params"));
        return;
    }

    let text_document : Optional<JSON_Node> = req.params:data().map:at(sv("textDocument"));
    if !text_document:has_value() {
        print(sv("'textDocument' not found"));
        return;
    }

    let uri : Optional<JSON_Node> = text_document:data().map:at(sv("uri"));
    if !uri:has_value() {
        print(sv("'uri' not found"));
        return;
    }
    if !uri:data().value:has_value() {
        print(sv("Failed to parse uri"));
        return;
    }

    // We expect `file:///file/system/path`.
    let file_scheme : StringView = sv("file://");
    let protocol : StringView = uri:data().value:data():slice_to(file_scheme:length());
    if protocol != file_scheme {
        printf(sv("Unsupported URI protocol `%`"), protocol);
        return;
    }

    let path : StringView = uri:data().value:data():slice_from(7);
    mut json : Optional<JSON_Node> = state:run_parser(path);

    if !json:has_value() {
        // run_parser() already printed an appropriate error.
        return;
    }

    let parser_output : Optional<ParserOutput> = state:parse_parser_output(&mut json:data());

    if !parser_output:has_value() {
        json:data():deallocate();
        return;
    }

    if !parser_output:data().diagnostic:has_value() {
        // The parser didn't output an error, so the code is good. We still need
        // to reply though.
        let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{\"kind\":\"full\", \"items\": []}}");
        mut ss : StringStream = make<StringStream>(&mut state.allocator);
        ss:writef(resp, req.id:data());
        out:send_response(ss:as_sv());
        ss:deallocate();
        json:data():deallocate();
        return;
    }

    let diagnostic : Diagnostic = parser_output:data().diagnostic:data();

    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"id\":%,\"result\":{\"kind\":\"full\", \"items\": [{ \"range\": { \"start\": { \"line\": %, \"character\": % }, \"end\" : { \"line\": %, \"character\": % } }, \"message\": % }]}}");
    mut ss : StringStream = make<StringStream>(&mut state.allocator);
    ss:writef(
        resp,
        req.id:data(),
        diagnostic.line,
        diagnostic.column,
        diagnostic.line,
        diagnostic.column,
        diagnostic.message,
    );
    out:send_response(ss:as_sv());

    ss:deallocate();
    json:data():deallocate();
}

function handle_text_change : (
    state : State mut&, req : RPCNotification, out : BufferedFile mut&
) -> void = {
    printf(sv("Received textDocument/didChange notification"));

    if !req.params:has_value() {
        print(sv("Missing params"));
        return;
    }

    let text_document : Optional<JSON_Node> = req.params:data().map:at(sv("textDocument"));
    if !text_document:has_value() {
        print(sv("'textDocument' not found"));
        return;
    }

    let uri : Optional<JSON_Node> = text_document:data().map:at(sv("uri"));
    if !uri:has_value() {
        print(sv("'uri' not found"));
        return;
    }

    let version : Optional<JSON_Node> = text_document:data().map:at(sv("version"));
    if !version:has_value() {
        print(sv("'version' not found"));
        return;
    }

    let content_changes : Optional<JSON_Node> = req.params:data().map:at(sv("contentChanges"));
    if !content_changes:has_value() {
        print(sv("'contentChanges' not found"));
        return;
    }

    if content_changes:data().children:length() == 0 {
        print(sv("'contentChanges' is empty"));
        return;
    }

    let text_wrapper : JSON_Node = content_changes:data().children:get(0);
    let text : Optional<JSON_Node> = text_wrapper.map:at(sv("text"));
    if !text:has_value() {
        print(sv("'text' not found"));
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

    mut path : StringStream = make<StringStream>(&mut state.allocator);
    path:writef(sv("/proc/%/fd/%"), getpid(), fd);

    mut json : Optional<JSON_Node> = state:run_parser(path:as_sv());

    path:deallocate();
    munmap(addr, text_length);
    sys_close(fd);

    if !json:has_value() {
        // run_parser() already printed an appropriate error.
        return;
    }

    let parser_output : Optional<ParserOutput> = state:parse_parser_output(&mut json:data());

    if !parser_output:has_value() {
        json:data():deallocate();
        return;
    }

    if !parser_output:data().diagnostic:has_value() {
        // The parser didn't output an error, so the code is good. We still need
        // to reply though.
        let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{\"uri\": %, \"version\": %, \"diagnostics\": []}}");
        mut ss : StringStream = make<StringStream>(&mut state.allocator);
        ss:writef(resp, uri:data(), version:data());
        out:send_response(ss:as_sv());
        ss:deallocate();
        json:data():deallocate();
        return;
    }

    let diagnostic : Diagnostic = parser_output:data().diagnostic:data();

    let resp : StringView = sv("{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{\"uri\": %, \"version\": %, \"diagnostics\": [{ \"range\": { \"start\": { \"line\": %, \"character\": % }, \"end\" : { \"line\": %, \"character\": % } }, \"message\": % }]}}");
    mut ss : StringStream = make<StringStream>(&mut state.allocator);
    ss:writef(
        resp,
        uri:data(),
        version:data(),
        diagnostic.line,
        diagnostic.column,
        diagnostic.line,
        diagnostic.column,
        diagnostic.message,
    );
    out:send_response(ss:as_sv());

    ss:deallocate();
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
        return;
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

    if !saw_parser_arg {
        if argc >= 1 {
            printf(
                sv("Usage: % --parser=/path/to/parser [--pipe=/path/to/socket]"),
                make<CStringView>(&argv[0])
            );
        }
        return 1;
    }

    print(sv("Starting!"));

    mut state : State = make<State>(parser_path);

    while true {
        if !state:handle_next_rpc(in_file, out_file) {
            // Try and resync if something went wrong.
            state.client_buffer:clear();
        }
    }

    return 0;
}

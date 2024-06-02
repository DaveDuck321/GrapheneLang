@require_once "std/io.c3"
@require_once "std/memory.c3"
@require_once "std/format.c3"
@require_once "std/fcntl.c3"
@require_once "std/json.c3"
@require_once "std/sys/socket.c3"

@require_once "ring_buffer.c3"

typedef RPCRequest : {
    id : StringView, // TODO parse to int.
    method : StringView, // TODO string.
    params : JSON_Node, // TODO mut&?
}

function [IO] write : (io : IO mut&, req : RPCRequest) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    io:writef(sv("RPCRequest {id=%, method=%, params=%}"), req.id, req.method, req.params);
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
    fd : File, allocator : Allocator mut&
) -> Optional<RPCRequest> = {
    mut cb_opt : Optional<RingBuffer> = make<RingBuffer>(8192, {"ipc_buffer"});
    if !cb_opt:has_value() {
        print(sv("Failed to allocate ring buffer"));
        return make<Optional<RPCRequest>>();
    }

    mut cb : RingBuffer = cb_opt:data();

    mut view_opt : Optional<View<u8>> = cb:read_line(fd);
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

    cb:advance(view:length());
    view_opt = cb:read_line(fd);
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

    cb:advance(view:length());
    view_opt = cb:read(fd, length:data());
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

    cb:advance(view:length());

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

function dispatch_rpc_request : (req : RPCRequest, fd : File) -> void = {
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

    while true {
        let parsed : Optional<RPCRequest> = parse_next_rpc(in_file, &mut allocator);

        if parsed:has_value() {
            dispatch_rpc_request(parsed:data(), out_file);
        }
    }

    return 0;
}

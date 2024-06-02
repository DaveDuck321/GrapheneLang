// sys/socket.h - main sockets header
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/socket.h.html
@require_once "../io.c3"
@require_once "../util.c3"
@require_once "syscalls.c3"

typedef sa_family_t : u16

// Structure describing the address of an AF_UNIX socket.
typedef sockaddr_un : {
    sun_family : sa_family_t,
    sun_path : u8[108],
}

function AF_UNIX : () -> int = {
    return 1;
}

function SOCK_STREAM : () -> int = {
    return 1;
}

function socket : (domain : int, type : int, protocol : int) -> Optional<File> = {
    let fd : int = Narrow<int>(sys_socket(domain, type, protocol));
    if fd < 0 {
        return make<Optional<File>>();
    }
    return make<Optional<File>>({ fd });
}

function connect : (
    socket : File, address : sockaddr_un&
) -> int = {
    // Need to work out the real size of the address.
    let path : CString = { &address.sun_path };
    let size : isize = sizeof<sa_family_t>() + path:length();
    return Narrow<int>(sys_connect(socket.fd, ref_to_addr(&address), Narrow<int>(size)));
}

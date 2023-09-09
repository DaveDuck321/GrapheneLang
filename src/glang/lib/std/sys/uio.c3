// sys/uio.h - definitions for vector I/O operations
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/uio.h.html

typedef iovec : {
    iov_base : u8[&], // Base address of a memory region for input or output.
    iov_len  : isize  // The size of the memory pointed to by iov_base.
}

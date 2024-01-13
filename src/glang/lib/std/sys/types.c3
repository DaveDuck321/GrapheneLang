// sys/types.h - data types
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/types.h.html

typedef blkcnt_t : isize
typedef blksize_t : isize
typedef dev_t : u64
typedef gid_t : id_t
typedef id_t : u32
typedef ino_t : u64
typedef mode_t : u32
typedef nlink_t : u64
typedef off_t : isize
typedef pid_t : int
typedef time_t : isize
typedef uid_t : id_t

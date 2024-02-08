// sys/stat.h - data returned by the stat() function
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/stat.h.html
@require_once "../wrappers.c3"
@require_once "syscalls.c3"
@require_once "types.c3"
@require_once "sys/kstat.c3"  // arch-specific `stat_t` definition

function stat : ( fd : int ) -> Optional<stat_t> = {
    // TODO: propagate the actual error code
    mut stat_buf : stat_t;
    let result : int = sys_fstat(fd, &mut stat_buf);
    if result < 0 {
        return make<Optional<stat_t>>();
    }
    return make<Optional<stat_t>>(stat_buf);
}

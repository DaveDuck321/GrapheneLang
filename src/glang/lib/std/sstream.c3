@require_once "string.c3"
@require_once "memory.c3"
@require_once "vector.c3"

typedef RegisterIsStreamableTextIO<Override<StringStream>> : TrueType
typedef StringStream : {
    buffer : Vector<u8>,
    buffer_pointer : isize,
}

function make<StringStream> : (allocator : Allocator mut&) -> StringStream = {
    return {
        .buffer = make<Vector<u8>>(&mut allocator),
        .buffer_pointer = 0,
    };
}

function length : (ss : StringStream&) -> isize = {
    return ss.buffer_pointer;
}

function as_sv : (self : StringStream&) -> StringView = {
    let underlying : View<u8> = self.buffer:view();
    return make<StringView>(underlying:slice_to(self:length()));
}

function flush : (self : StringStream mut&) -> void = {
    // Nothing to do.
}

function close : (self : StringStream mut&) -> void = {
    // Nothing to do.
}

function request_buffer : (self : StringStream mut&, length : isize) -> Span<u8> = {
    self.buffer:resize(self:length() + length, char("\0"));
    let span : Span<u8> = self.buffer:data();
    return span:slice_from(self:length());
}

function consume_length : (self : StringStream mut&, length : isize) -> void = {
    self.buffer_pointer += length;
}

function deallocate : (self : StringStream mut&) -> void = {
    self.buffer:deallocate();
    self.buffer_pointer = 0;
}

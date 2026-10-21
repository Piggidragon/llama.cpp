#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // whether write_tensor_snap() can be used (LLAMA_STATE_SEQ_FLAGS_SNAPSHOT)
    virtual bool can_snapshot() const { return false; }

    // like write_tensor(), but the data is copied to snap at snap_offset (a device snapshot) instead: of the size bytes
    // in the buffer, the first n_tag are tag and the others stay undefined
    virtual void write_tensor_snap(
            ggml_tensor * tensor, size_t offset, size_t size, ggml_tensor * snap, size_t snap_offset, const void * tag, size_t n_tag);

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // the next size bytes without reading them, nullptr if they cannot be seen in advance
    virtual const uint8_t * peek(size_t size) { (void) size; return nullptr; }

    // like read_tensor(), but the data is copied from snap at snap_offset (a device snapshot): the size bytes of the
    // buffer are skipped
    virtual void read_tensor_snap(ggml_tensor * tensor, size_t offset, size_t size, ggml_tensor * snap, size_t snap_offset);

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};

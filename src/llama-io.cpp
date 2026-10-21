#include "llama-io.h"

#include <stdexcept>
#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

void llama_io_write_i::write_tensor_snap(
        ggml_tensor * /*tensor*/, size_t /*offset*/, size_t /*size*/, ggml_tensor * /*snap*/, size_t /*snap_offset*/, const void * /*tag*/, size_t /*n_tag*/) {
    throw std::runtime_error("this writer cannot keep tensor data in a device snapshot");
}

void llama_io_read_i::read_tensor_snap(ggml_tensor * /*tensor*/, size_t /*offset*/, size_t /*size*/, ggml_tensor * /*snap*/, size_t /*snap_offset*/) {
    throw std::runtime_error("this reader cannot read tensor data from a device snapshot");
}

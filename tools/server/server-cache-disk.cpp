#include "server-cache-disk.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <unordered_map>

#if defined(_WIN32)
// windows.h defines min and max as macros, which breaks std::min and std::max
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// a whole state, the head of a ranged state, and a KV block of a ranged state
static constexpr uint32_t DISK_MAGIC       = 0x44435053; // "SPCD"
static constexpr uint32_t DISK_MAGIC_HEAD  = 0x48435053; // "SPCH"
static constexpr uint32_t DISK_MAGIC_BLOCK = 0x4b435053; // "SPCK"
static constexpr uint32_t DISK_VERSION     = 1;
static constexpr uint32_t DISK_VERSION_KV  = 2; // heads and blocks: 128-bit SHA-256 block keys

// blobs are written in chunks, so that fetch() can stop a write early
static constexpr size_t DISK_CHUNK = 64ull*1024*1024;

// free space the disk level leaves on the file system: 5% of it, at least 1 GiB
static size_t disk_reserve(const fs::space_info & space) {
    return std::max<size_t>(1024ull*1024*1024, space.capacity / 20);
}

static constexpr double MiB = 1024.0*1024.0;

static uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const auto * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

namespace {

// FILE wrapper that counts bytes and remembers the first error
struct disk_file {
    FILE * f  = nullptr;
    bool   ok = false;
    size_t n  = 0; // bytes read or written

    disk_file(const std::string & path, const char * mode) : f(ggml_fopen(path.c_str(), mode)), ok(f != nullptr) {}

    ~disk_file() {
        if (f) {
            fclose(f);
        }
    }

    void write(const void * src, size_t size) {
        if (ok && size > 0) {
            ok = fwrite(src, 1, size, f) == size;
            n += size;
        }
    }

    void read(void * dst, size_t size) {
        if (ok && size > 0) {
            ok = fread(dst, 1, size, f) == size;
            n += size;
        }
    }

    template <typename T> void write_val(const T & val) {
        write(&val, sizeof(val));
    }

    template <typename T> T read_val() {
        T val{};
        read(&val, sizeof(val));
        return val;
    }

    bool seek(uint64_t offset) {
#if defined(_WIN32)
        return _fseeki64(f, (__int64) offset, SEEK_SET) == 0;
#else
        return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
    }

    // flush to the device, so that a renamed file is complete after a crash
    bool sync() {
        if (!ok || fflush(f) != 0) {
            return false;
        }
#if defined(_WIN32)
        return _commit(_fileno(f)) == 0;
#else
        return fsync(fileno(f)) == 0;
#endif
    }
};

}

static bool write_blob(disk_file & f, const std::vector<uint8_t> & buf, const std::atomic<bool> & cancel) {
    f.write_val<uint64_t>(buf.size());
    for (size_t off = 0; off < buf.size() && f.ok; off += DISK_CHUNK) {
        if (cancel.load()) {
            return false;
        }
        f.write(buf.data() + off, std::min(DISK_CHUNK, buf.size() - off));
    }
    return f.ok;
}

static bool write_blob(disk_file & f, const common_bytes & buf, const std::atomic<bool> & cancel) {
    f.write_val<uint64_t>(buf.size());
    for (size_t off = 0; off < buf.size() && f.ok; off += DISK_CHUNK) {
        if (cancel.load()) {
            return false;
        }
        f.write(buf.data() + off, std::min(DISK_CHUNK, buf.size() - off));
    }
    return f.ok;
}

static bool read_blob(disk_file & f, common_bytes & buf, size_t file_size) {
    const uint64_t size = f.read_val<uint64_t>();
    // check the size against the bytes left before allocating, the field may be corrupted
    if (!f.ok || size > file_size - f.n) {
        return false;
    }
    buf.resize(size);
    f.read(buf.wdata(), size);
    return f.ok;
}

static bool read_blob(disk_file & f, std::vector<uint8_t> & buf, size_t file_size) {
    const uint64_t size = f.read_val<uint64_t>();
    // check the size against the bytes left before allocating, the field may be corrupted
    if (!f.ok || size > file_size - f.n) {
        return false;
    }
    buf.resize(size);
    f.read(buf.data(), size);
    return f.ok;
}

static std::string hex64(uint64_t val) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, val);
    return buf;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//
// server_prompt_cache_disk::file_lock
//

server_prompt_cache_disk::file_lock::file_lock(file_lock && other) noexcept : handle(other.handle) {
    other.handle = -1;
}

server_prompt_cache_disk::file_lock & server_prompt_cache_disk::file_lock::operator=(file_lock && other) noexcept {
    if (this != &other) {
        release();
        handle       = other.handle;
        other.handle = -1;
    }
    return *this;
}

server_prompt_cache_disk::file_lock::~file_lock() {
    release();
}

void server_prompt_cache_disk::file_lock::release() {
    if (handle == -1) {
        return;
    }
#if defined(_WIN32)
    CloseHandle((HANDLE) handle);
#else
    ::close((int) handle);
#endif
    handle = -1;
}

server_prompt_cache_disk::file_lock::result server_prompt_cache_disk::file_lock::try_lock(const std::string & path) {
    release();

#if defined(_WIN32)
    HANDLE h = CreateFileW(fs::u8path(path).wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return result::failed;
    }

    OVERLAPPED ov = {};
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) {
        const DWORD err = GetLastError();
        CloseHandle(h);
        return err == ERROR_LOCK_VIOLATION ? result::busy : result::failed;
    }

    handle = (intptr_t) h;
#else
    // not inherited: a child process would hold the lock as long as it runs
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        return result::failed;
    }

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        const int err = errno;
        ::close(fd);
        return err == EWOULDBLOCK ? result::busy : result::failed;
    }

    handle = fd;
#endif

    return result::locked;
}

//
// server_prompt_cache_disk
//

std::string server_prompt_cache_disk::lock_path(const std::string & dir, uint64_t fingerprint) {
    return fs_path_to_utf8(fs::u8path(dir) / fs::u8path("pc-" + hex64(fingerprint) + ".lock"));
}

std::unique_ptr<server_prompt_cache_disk> server_prompt_cache_disk::create(const std::string & dir, size_t limit_size, size_t limit_pending, uint64_t fingerprint, bool has_mtmd) {
    std::error_code ec;
    fs::create_directories(fs::u8path(dir), ec);

    const std::string path = lock_path(dir, fingerprint);

    // a server of another fingerprint holds the lock while it removes files of this one, a server of this
    // fingerprint that is stopping holds it until it has written its last states: wait for them a while
    file_lock lock;
    auto res = lock.try_lock(path);
    for (int i = 0; res == file_lock::result::busy && i < 100; ++i) {
        if (i == 10) {
            SRV_INF("disk prompt cache: waiting for another server that uses '%s' with the same model and settings\n", dir.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res = lock.try_lock(path);
    }

    if (res == file_lock::result::busy) {
        SRV_WRN("disk prompt cache disabled: another server uses '%s' with the same model and settings\n", dir.c_str());
        return nullptr;
    }

    if (res == file_lock::result::failed) {
        SRV_WRN("disk prompt cache: cannot lock '%s', a second server with the same model and settings would remove the files of this one\n", path.c_str());
    }

    return std::make_unique<server_prompt_cache_disk>(dir, limit_size, limit_pending, fingerprint, has_mtmd, std::move(lock));
}

server_prompt_cache_disk::server_prompt_cache_disk(const std::string & dir, size_t limit_size, size_t limit_pending, uint64_t fingerprint, bool has_mtmd, file_lock && lock)
    : dir(dir), prefix("pc-" + hex64(fingerprint) + "-"), lock_own(std::move(lock)),
      limit_size(limit_size), limit_pending(limit_pending), fingerprint(fingerprint), has_mtmd(has_mtmd) {
    scan();

    worker = std::thread([this]() { worker_loop(); });
}

server_prompt_cache_disk::~server_prompt_cache_disk() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        stopping = true;

        if (!queue.empty()) {
            SRV_INF("disk prompt cache: writing %zu remaining prompt(s) (%.3f MiB) before exit\n", queue.size(), size_pending / MiB);
        }
    }
    cv.notify_all();

    if (worker.joinable()) {
        worker.join();
    }
}

std::list<server_prompt_cache_disk::entry>::iterator server_prompt_cache_disk::find_entry(int64_t id) {
    return std::find_if(entries.begin(), entries.end(), [id](const entry & e) { return e.id == id; });
}

std::string server_prompt_cache_disk::block_path(const server_kv_key & key) const {
    return fs_path_to_utf8(fs::u8path(dir) / fs::u8path(prefix + "k" + hex64(key.a) + hex64(key.b) + ".bin"));
}

size_t server_prompt_cache_disk::size_on_disk() const {
    size_t res = 0;
    for (const auto & e : entries) {
        if (e.on_disk) {
            res += e.size;
        }
    }
    for (const auto & it : blocks) {
        if (it.second.on_disk) {
            res += it.second.size;
        }
    }
    return res;
}

size_t server_prompt_cache_disk::size() const {
    std::lock_guard<std::mutex> lk(mtx);
    return size_on_disk();
}

size_t server_prompt_cache_disk::n_entries() const {
    std::lock_guard<std::mutex> lk(mtx);
    return std::count_if(entries.begin(), entries.end(), [](const entry & e) { return !e.foreign; });
}

void server_prompt_cache_disk::release_blocks(const entry & e) {
    for (const auto & key : e.blocks) {
        auto it = blocks.find({ e.fp, key });
        if (it == blocks.end() || --it->second.refs > 0) {
            continue;
        }

        // no head holds the block any more
        if (it->second.on_disk) {
            std::error_code ec;
            fs::remove(fs::u8path(it->second.path), ec);
        }

        blocks.erase(it);
    }
}

void server_prompt_cache_disk::erase_entry(std::list<entry>::iterator it) {
    GGML_ASSERT(!it->busy());

    if (it->on_disk) {
        std::error_code ec;
        fs::remove(fs::u8path(it->path), ec);
    }

    release_blocks(*it);

    if (it->pending) {
        size_pending -= it->size_mem;
        it->pending.reset();
    }

    queue.erase(std::remove(queue.begin(), queue.end(), it->id), queue.end());

    entries.erase(it);
}

void server_prompt_cache_disk::drop_superseded(const entry & cur) {
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);

        // a waiting entry is left alone: fetch() may wait for it after a cancelled write
        if (&*it != &cur && !it->busy() && !it->pending && !it->foreign && it->tokens.size() <= cur.tokens.size() &&
            cur.tokens.get_common_prefix(it->tokens) == it->tokens.size()) {
            SRV_TRC("disk prompt cache: removing prompt with %zu tokens, contained in the new one\n", it->tokens.size());
            erase_entry(it);
        }

        it = next;
    }
}

bool server_prompt_cache_disk::make_room(size_t size_new, int64_t id_keep, time_point t_new) {
    // the servers of the fingerprints that refresh_foreign() locked may start once their files are removed
    struct release_locks {
        std::vector<file_lock> & locks;
        ~release_locks() { locks.clear(); }
    } release { locks_foreign };

    if (limit_size > 0 && size_new > limit_size) {
        return false;
    }

    // bytes to free for the size limit, and to keep the reserve free on the file system
    auto bytes_needed = [&]() {
        size_t need = 0;

        const size_t size_cur = size_on_disk();
        if (limit_size > 0 && size_cur + size_new > limit_size) {
            need = size_cur + size_new - limit_size;
        }

        if (size_new > 0) {
            std::error_code ec;
            const auto space = fs::space(fs::u8path(dir), ec);
            const size_t reserve = ec ? 0 : disk_reserve(space);
            if (!ec && space.available < size_new + reserve) {
                need = std::max<size_t>(need, size_new + reserve - space.available);
            }
        }

        return need;
    };

    size_t need = bytes_needed();
    if (need == 0) {
        return true;
    }

    // another server may have written heads that share blocks of its fingerprint since the last look
    refresh_foreign();

    need = bytes_needed();
    if (need == 0) {
        return true;
    }

    // least recently used files first, and only files used before the new state
    std::vector<std::list<entry>::iterator> lru;
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->on_disk && !it->busy() && !it->pending && it->id != id_keep && it->t_used < t_new) {
            lru.push_back(it);
        }
    }
    std::sort(lru.begin(), lru.end(), [](const auto & a, const auto & b) { return a->t_used < b->t_used; });

    // an entry frees its file and the blocks that no other entry holds
    // the bytes are counted from the file sizes: a copy-on-write file system can report the free space later
    std::unordered_map<block_id, int, block_id_hash> refs;

    size_t n_free  = 0;
    size_t n_evict = 0;
    while (n_evict < lru.size() && n_free < need) {
        const entry & e = *lru[n_evict++];

        n_free += e.size;

        for (const auto & key : e.blocks) {
            const block_id id { e.fp, key };

            auto it = blocks.find(id);
            if (it == blocks.end()) {
                continue;
            }

            auto res = refs.emplace(id, it->second.refs);
            if (--res.first->second == 0 && it->second.on_disk) {
                n_free += it->second.size;
            }
        }
    }

    SRV_TRC("disk prompt cache: %.3f MiB needed, %zu file(s) free %.3f MiB\n", need / MiB, n_evict, n_free / MiB);

    if (n_free < need) {
        return false;
    }

    for (size_t i = 0; i < n_evict; ++i) {
        SRV_INF("disk prompt cache: removing least recently used %s with %zu tokens (%.3f MiB)\n",
                lru[i]->foreign ? "file of another fingerprint" : "prompt", lru[i]->tokens.size(), lru[i]->size / MiB);

        erase_entry(lru[i]);
    }

    return true;
}

void server_prompt_cache_disk::refresh_foreign() {
    // forget what is known: another server may have written or removed files since
    entries.remove_if([](const entry & e) { return e.foreign; });
    for (auto it = blocks.begin(); it != blocks.end();) {
        if (it->first.fp != fingerprint) {
            it = blocks.erase(it);
        } else {
            ++it;
        }
    }

    locks_foreign.clear();

    // a fingerprint whose lock is held has a running server, which may hold its blocks in memory, in a queued
    // state or in a head it is writing. the lock of any other fingerprint is kept, so that its server does not
    // start while its files are removed
    std::unordered_map<uint64_t, bool> live;
    auto is_live = [&](uint64_t fp) {
        auto res = live.emplace(fp, true);
        if (res.second) {
            file_lock lock;
            if (lock.try_lock(lock_path(dir, fp)) == file_lock::result::locked) {
                res.first->second = false;
                locks_foreign.push_back(std::move(lock));
            }
        }
        return res.first->second;
    };

    std::error_code ec;

    try {
        for (const auto & de : fs::directory_iterator(fs::u8path(dir))) {
            const std::string name = fs_path_to_utf8(de.path().filename());
            const std::string path = fs_path_to_utf8(de.path());

            if (name.rfind("pc-", 0) != 0 || name.size() <= prefix.size() || name[prefix.size() - 1] != '-' ||
                !ends_with(name, ".bin") || !de.is_regular_file(ec)) {
                continue;
            }

            uint64_t fp = 0;
            try {
                fp = std::stoull(name.substr(3, 16), nullptr, 16);
            } catch (...) {
                continue;
            }

            if (fp == fingerprint) {
                continue;
            }

            is_live(fp);

            const char kind = name[prefix.size()];

            if (kind == 'k') {
                server_kv_block block;
                if (read_block(path, fp, block, true)) {
                    auto & b = blocks[{ fp, block.key }];
                    b.path    = path;
                    b.size    = de.file_size(ec);
                    b.on_disk = true;
                    continue;
                }
            }

            // a head with its block list; a whole state, or a file of a format this version does not read, is just a file
            server_prompt_cache_state  state;
            std::vector<server_kv_key> keys;

            const bool ranged = kind == 'h' && read_head(path, fp, state, keys, true);

            entries.emplace_back();

            auto & e = entries.back();
            e.id      = id_next++;
            e.path    = path;
            e.fp      = fp;
            e.ranged  = ranged;
            e.blocks  = std::move(keys);
            e.size    = de.file_size(ec);
            e.t_used  = de.last_write_time(ec);
            e.on_disk = true;
            e.foreign = true;
        }
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: failed to scan '%s': %s\n", dir.c_str(), err.what());
    }

    // references of the blocks; a block of a missing head is not counted, another server may still be writing it
    for (auto & e : entries) {
        if (!e.foreign) {
            continue;
        }

        const uint64_t fp = e.fp;

        if (live.at(fp)) {
            // removing the head leaves its blocks alone
            e.blocks.clear();
            continue;
        }

        e.blocks.erase(std::remove_if(e.blocks.begin(), e.blocks.end(), [&](const server_kv_key & key) { return blocks.count({ fp, key }) == 0; }), e.blocks.end());

        for (const auto & key : e.blocks) {
            blocks[{ fp, key }].refs++;
        }
    }

    // blocks that no head holds become files that the LRU can evict
    for (auto it = blocks.begin(); it != blocks.end();) {
        if (it->first.fp == fingerprint || it->second.refs > 0 || live.at(it->first.fp)) {
            ++it;
            continue;
        }

        entries.emplace_back();

        auto & e = entries.back();
        e.id      = id_next++;
        e.path    = it->second.path;
        e.fp      = it->first.fp;
        e.size    = it->second.size;
        e.t_used  = fs::last_write_time(fs::u8path(it->second.path), ec);
        e.on_disk = true;
        e.foreign = true;

        it = blocks.erase(it);
    }
}

void server_prompt_cache_disk::scan() {
    std::error_code ec;
    fs::create_directories(fs::u8path(dir), ec);
    if (ec) {
        SRV_WRN("disk prompt cache: cannot create '%s': %s\n", dir.c_str(), ec.message().c_str());
        return;
    }

    try {
        for (const auto & de : fs::directory_iterator(fs::u8path(dir))) {
            const std::string name = fs_path_to_utf8(de.path().filename());
            const std::string path = fs_path_to_utf8(de.path());

            // pc-<16 hex digits of the fingerprint>-<kind and hash>
            if (name.rfind("pc-", 0) != 0 || name.size() <= prefix.size() || name[prefix.size() - 1] != '-' || !de.is_regular_file()) {
                continue;
            }

            uint64_t fp = 0;
            try {
                fp = std::stoull(name.substr(3, 16), nullptr, 16);
            } catch (...) {
                continue;
            }

            const bool own = fp == fingerprint;
            if (!own) {
                continue;
            }

            if (ends_with(name, ".tmp")) {
                // left behind by an interrupted write
                fs::remove(de.path(), ec);
                continue;
            }

            if (!ends_with(name, ".bin")) {
                continue;
            }

            // a KV block: 'k', a head of a ranged state: 'h', a whole state: hex digits
            const char kind = name[prefix.size()];

            if (kind == 'k') {
                server_kv_block block;
                if (read_block(path, fp, block, true)) {
                    auto & b = blocks[{ fp, block.key }];
                    b.path    = path;
                    b.size    = de.file_size();
                    b.on_disk = true;
                } else {
                    SRV_WRN("disk prompt cache: removing unreadable file '%s'\n", path.c_str());
                    fs::remove(de.path(), ec);
                }
                continue;
            }

            server_prompt_cache_state  state;
            std::vector<server_kv_key> keys;

            const bool ok = kind == 'h' ? read_head(path, fp, state, keys, true) : read_file(path, state, true);

            if (!ok) {
                SRV_WRN("disk prompt cache: removing unreadable file '%s'\n", path.c_str());
                fs::remove(de.path(), ec);
                continue;
            }

            entries.emplace_back();

            auto & e = entries.back();
            e.id      = id_next++;
            e.path    = path;
            e.fp      = fp;
            e.ranged  = kind == 'h';
            e.blocks  = std::move(keys);
            e.tokens  = std::move(state.prompt.tokens);
            e.size    = de.file_size();
            e.t_used  = de.last_write_time();
            e.on_disk = true;
        }
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: failed to scan '%s': %s\n", dir.c_str(), err.what());
    }

    // a head whose blocks are missing cannot be restored
    for (auto it = entries.begin(); it != entries.end();) {
        auto missing = [&](const server_kv_key & key) { return blocks.count({ fingerprint, key }) == 0; };

        if (std::any_of(it->blocks.begin(), it->blocks.end(), missing)) {
            SRV_WRN("disk prompt cache: removing '%s', a KV block of it is missing\n", it->path.c_str());
            fs::remove(fs::u8path(it->path), ec);
            it = entries.erase(it);
            continue;
        }

        for (const auto & key : it->blocks) {
            blocks[{ fingerprint, key }].refs++;
        }

        ++it;
    }

    // blocks that no head holds, left behind by an interrupted write
    for (auto it = blocks.begin(); it != blocks.end();) {
        if (it->second.refs == 0) {
            fs::remove(fs::u8path(it->second.path), ec);
            it = blocks.erase(it);
        } else {
            ++it;
        }
    }

    refresh_foreign();

    make_room(0, -1, time_point::max());

    const size_t n_own        = std::count_if(entries.begin(), entries.end(), [](const entry & e) { return !e.foreign; });
    const size_t n_own_blocks = std::count_if(blocks.begin(), blocks.end(), [this](const auto & it) { return it.first.fp == fingerprint; });

    SRV_INF("disk prompt cache: %zu prompt(s) with %zu KV block(s), %zu file(s) of other fingerprints, %.3f MiB in '%s' (limit: %.3f MiB)\n",
            n_own, n_own_blocks, entries.size() - n_own + blocks.size() - n_own_blocks, size_on_disk() / MiB, dir.c_str(), limit_size / MiB);
}

void server_prompt_cache_disk::put(server_prompt_cache_state && state, time_point t_used, bool bounded) {
    const size_t n_tokens = state.prompt.tokens.size();
    if (n_tokens == 0) {
        return;
    }

    std::vector<char> packed;
    try {
        packed = state.prompt.tokens.serialize();
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: cannot store prompt with %zu tokens: %s\n", n_tokens, err.what());
        return;
    }

    const bool ranged = state.data.ranged;

    const std::string name = prefix + (ranged ? "h" : "") + hex64(fnv1a(FNV_OFFSET, packed.data(), packed.size())) + ".bin";

    std::lock_guard<std::mutex> lk(mtx);

    for (const auto & e : entries) {
        if (e.tokens.size() >= n_tokens && e.tokens.get_common_prefix(state.prompt.tokens) == n_tokens) {
            SRV_TRC("disk prompt cache: prompt with %zu tokens is already stored, skipping\n", n_tokens);
            return;
        }
    }

    // waiting states that this one continues would be removed once it is written, drop them now
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (!it->on_disk && !it->busy() && state.prompt.tokens.get_common_prefix(it->tokens) == it->tokens.size()) {
            erase_entry(it);
        }
        it = next;
    }

    // bytes the state adds to the queue: its own data and the blocks that are neither on disk nor waiting
    size_t size_state = state.size_own();
    for (const auto & block : state.data.blocks) {
        auto it = blocks.find({ fingerprint, block->key });
        if (it == blocks.end() || (!it->second.on_disk && !it->second.data)) {
            size_state += block->size();
        }
    }

    // the disk is behind: drop the oldest waiting states to bound the memory they hold
    while (bounded && limit_pending > 0 && size_pending + size_state > limit_pending) {
        auto it_old = entries.end();
        for (const int64_t id : queue) {
            auto it = find_entry(id);
            if (it != entries.end() && !it->busy()) {
                it_old = it;
                break;
            }
        }

        if (it_old == entries.end()) {
            break;
        }

        SRV_WRN("disk prompt cache: writes are behind, dropping waiting prompt with %zu tokens (%.3f MiB)\n",
                it_old->tokens.size(), it_old->size_mem / MiB);

        erase_entry(it_old);
    }

    entries.emplace_back();

    auto & e = entries.back();
    e.id       = id_next++;
    e.path     = fs_path_to_utf8(fs::u8path(dir) / fs::u8path(name));
    e.fp       = fingerprint;
    e.tokens   = state.prompt.tokens.clone();
    e.ranged   = ranged;
    e.size     = size_state;
    e.size_mem = size_state;
    e.t_used   = t_used;

    for (const auto & block : state.data.blocks) {
        e.blocks.push_back(block->key);

        auto & b = blocks[{ fingerprint, block->key }];
        b.refs++;
        if (!b.on_disk && !b.data) {
            b.path = block_path(block->key);
            b.data = block;
        }
    }

    e.pending = std::make_unique<server_prompt_cache_state>(std::move(state));

    queue.push_back(e.id);
    size_pending += size_state;

    SRV_TRC("disk prompt cache: queued prompt with %zu tokens (%.3f MiB new), %zu waiting (%.3f MiB)\n",
            n_tokens, size_state / MiB, queue.size(), size_pending / MiB);

    cv.notify_all();
}

bool server_prompt_cache_disk::contains(const server_tokens & tokens) const {
    std::lock_guard<std::mutex> lk(mtx);

    for (const auto & e : entries) {
        if (e.tokens.size() >= tokens.size() && e.tokens.get_common_prefix(tokens) == tokens.size()) {
            return true;
        }
    }

    return false;
}

int64_t server_prompt_cache_disk::find(const server_tokens & tokens_new, size_t n_tokens_max, size_t & lcp_best, bool whole_only) const {
    std::lock_guard<std::mutex> lk(mtx);

    int64_t id_best = -1;

    for (const auto & e : entries) {
        if (e.foreign || e.n_fail >= 2 || e.tokens.size() > n_tokens_max) {
            continue;
        }

        const size_t lcp_cur = e.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / e.tokens.size();

        SRV_TRC("   - disk prompt with length %7zu, lcp = %7zu, f_keep = %.3f\n", e.tokens.size(), lcp_cur, f_keep_cur);

        // same rule as the RAM level
        // a head is stored without checkpoints, a whole state keeps its checkpoints
        if (f_keep_cur < 0.25f || (whole_only && e.ranged && lcp_cur < e.tokens.size())) {
            continue;
        }

        if (lcp_cur > lcp_best) {
            lcp_best = lcp_cur;
            id_best  = e.id;
        }
    }

    return id_best;
}

bool server_prompt_cache_disk::fetch(int64_t id, server_prompt_cache_state & out, const block_source & source) {
    std::unique_lock<std::mutex> lk(mtx);

    auto it = find_entry(id);
    if (it == entries.end()) {
        return false;
    }

    if (it->writing) {
        // stop the write at the next chunk, the writer keeps the state for us
        it->cancel = true;
        cv.wait(lk, [&]() {
            auto cur = find_entry(id);
            return cur == entries.end() || !cur->writing;
        });

        it = find_entry(id);
        if (it == entries.end()) {
            return false;
        }
    }

    if (it->pending) {
        SRV_INF("disk prompt cache: taking back prompt with %zu tokens before it is written\n", it->tokens.size());

        out = std::move(*it->pending);

        it->pending.reset();
        size_pending -= it->size_mem;
        it->cancel = false;

        queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());

        if (!it->on_disk) {
            release_blocks(*it);
            entries.erase(it);
        }

        cv.notify_all();

        return true;
    }

    it->reading = true;

    const std::string path   = it->path;
    const bool        ranged = it->ranged;

    // the blocks cannot be removed while the entry is being read, it holds them
    std::vector<std::pair<server_kv_key, std::string>> block_files;
    for (const auto & key : it->blocks) {
        block_files.emplace_back(key, blocks.at({ fingerprint, key }).path);
    }

    lk.unlock();

    const int64_t t_start = ggml_time_us();

    bool          ok            = false;
    bool          corrupt       = false;
    bool          block_corrupt = false;
    server_kv_key key_corrupt;
    size_t        n_read        = 0;

    try {
        if (!ranged) {
            ok = read_file(path, out, false);
            corrupt = !ok;
        } else {
            std::vector<server_kv_key> keys;
            ok = read_head(path, fingerprint, out, keys, false);
            corrupt = !ok;

            // the head lists the same blocks as the index, in position order without gaps; a block of
            // another head that failed to read is not on disk any more
            size_t n_end = 0;
            for (size_t i = 0; ok && i < block_files.size(); ++i) {
                const server_kv_key & key = block_files[i].first;

                server_kv_block_ptr block = source ? source(key) : nullptr;
                if (!block) {
                    auto block_new = std::make_shared<server_kv_block>();
                    if (!read_block(block_files[i].second, fingerprint, *block_new, false) || !(block_new->key == key)) {
                        ok            = false;
                        corrupt       = true;
                        block_corrupt = true;
                        key_corrupt   = key;
                        break;
                    }
                    n_read += block_new->size();
                    block = std::move(block_new);
                }

                if (i >= keys.size() || !(keys[i] == key) || block->n_begin != n_end) {
                    ok      = false;
                    corrupt = true;
                    break;
                }
                n_end = block->n_end;

                out.data.blocks.push_back(std::move(block));
            }

            ok = ok && keys.size() == block_files.size() && n_end <= out.prompt.tokens.size();
            corrupt = corrupt || !ok;
        }
    } catch (const std::exception & err) {
        ok      = false;
        corrupt = false;
        SRV_WRN("disk prompt cache: failed to read '%s': %s\n", path.c_str(), err.what());
    }

    lk.lock();

    it->reading = false;

    if (!ok) {
        if (block_corrupt) {
            // the other heads that hold the block fail when they are read
            // a new state that holds the block writes it again
            auto bit = blocks.find({ fingerprint, key_corrupt });
            if (bit != blocks.end() && bit->second.on_disk) {
                SRV_WRN("disk prompt cache: KV block '%s' is corrupt, removing it\n", bit->second.path.c_str());
                std::error_code ec;
                fs::remove(fs::u8path(bit->second.path), ec);
                bit->second.on_disk = false;
                bit->second.size    = 0;
            }
        }
        if (corrupt) {
            SRV_WRN("disk prompt cache: '%s' is corrupt, removing it\n", path.c_str());
            erase_entry(it);
        }
        out = {};
        cv.notify_all();
        return false;
    }

    it->t_used = time_point::clock::now();

    std::error_code ec;
    fs::last_write_time(fs::u8path(path), it->t_used, ec);

    SRV_INF("disk prompt cache: read prompt with %zu tokens (%.3f MiB, %zu KV block(s)) in %.2f s\n",
            it->tokens.size(), (it->size + n_read) / MiB, it->blocks.size(), (ggml_time_us() - t_start) / 1e6);

    cv.notify_all();

    return true;
}

void server_prompt_cache_disk::restore_failed(int64_t id, server_prompt_cache_state && state, bool permanent) {
    {
        std::lock_guard<std::mutex> lk(mtx);

        auto it = find_entry(id);
        if (it != entries.end() && it->on_disk) {
            it->n_fail = permanent ? 2 : it->n_fail + 1;

            const bool remove = permanent && it->ranged && !it->busy();

            SRV_WRN("disk prompt cache: prompt with %zu tokens did not restore (%d time(s))%s\n",
                    it->tokens.size(), it->n_fail, remove ? ", removing it and its KV blocks" : it->n_fail < 2 ? "" : ", not using it any more");

            // a broken head may hold blocks of another state format: they are removed too, so that the prompts
            // that share them write them again instead of listing them
            // a failure for a lack of cells says nothing about the blocks, other heads may share them
            if (remove) {
                for (const auto & key : it->blocks) {
                    auto bit = blocks.find({ fingerprint, key });
                    if (bit != blocks.end() && bit->second.on_disk) {
                        std::error_code ec;
                        fs::remove(fs::u8path(bit->second.path), ec);
                        bit->second.on_disk = false;
                        bit->second.size    = 0;
                    }
                }

                erase_entry(it);
            }

            return;
        }
    }

    // it was taken back from the queue, write it after all
    put(std::move(state));
}

void server_prompt_cache_disk::flush() {
    std::unique_lock<std::mutex> lk(mtx);

    cv.wait(lk, [this]() {
        return queue.empty() && std::none_of(entries.begin(), entries.end(), [](const entry & e) { return e.writing; });
    });
}

void server_prompt_cache_disk::worker_loop() {
    std::unique_lock<std::mutex> lk(mtx);

    while (true) {
        cv.wait(lk, [this]() { return stopping || !queue.empty(); });

        if (queue.empty()) {
            // stopping, and nothing left to write
            break;
        }

        const int64_t id = queue.front();
        queue.pop_front();

        auto it = find_entry(id);
        if (it == entries.end() || !it->pending) {
            continue;
        }

        if (!make_room(it->size_mem, id, it->t_used)) {
            SRV_WRN("disk prompt cache: no room for prompt with %zu tokens (%.3f MiB), dropping it\n", it->tokens.size(), it->size_mem / MiB);
            erase_entry(it);
            cv.notify_all();
            continue;
        }

        it->writing = true;

        const std::string path     = it->path;
        const std::string path_tmp = path + ".tmp";
        const bool        ranged   = it->ranged;
        const time_point  t_used   = it->t_used;

        const server_prompt_cache_state & state = *it->pending;

        // the blocks to write: the worker is the only writer, so a block written for another entry is on disk already
        std::vector<std::pair<server_kv_key, server_kv_block_ptr>> todo;
        for (const auto & key : it->blocks) {
            const auto & b = blocks.at({ fingerprint, key });
            if (b.on_disk) {
                continue;
            }

            // the index has no data for a block that was on disk when the state was queued and was found
            // corrupt since then: the state holds it itself
            server_kv_block_ptr data = b.data;
            if (!data) {
                for (const auto & block : state.data.blocks) {
                    if (block->key == key) {
                        data = block;
                        break;
                    }
                }
            }

            todo.emplace_back(key, std::move(data));
        }

        lk.unlock();

        const int64_t t_start = ggml_time_us();

        bool ok = true;

        std::vector<std::pair<server_kv_key, size_t>> written; // key, file size
        size_t n_written = 0;

        std::error_code ec;

        try {
            for (const auto & [key, data] : todo) {
                const std::string block_file = block_path(key);
                const std::string block_tmp  = block_file + ".tmp";

                ok = data && write_block(block_tmp, *data, it->cancel);
                if (ok) {
                    fs::rename(fs::u8path(block_tmp), fs::u8path(block_file), ec);
                    ok = !ec;
                }
                if (!ok) {
                    fs::remove(fs::u8path(block_tmp), ec);
                    break;
                }

                const size_t size = fs::file_size(fs::u8path(block_file), ec);
                written.emplace_back(key, size);
                n_written += size;
            }

            if (ok) {
                ok = ranged ? write_head(path_tmp, state, it->cancel) : write_file(path_tmp, state, it->cancel);
            }
        } catch (const std::exception & err) {
            ok = false;
            SRV_WRN("disk prompt cache: failed to write '%s': %s\n", path_tmp.c_str(), err.what());
        }

        if (ok) {
            fs::rename(fs::u8path(path_tmp), fs::u8path(path), ec);
            ok = !ec;
        }
        if (ok) {
            // the mtime keeps the LRU order over restarts
            fs::last_write_time(fs::u8path(path), t_used, ec);
        } else {
            fs::remove(fs::u8path(path_tmp), ec);
        }

        const double t_s = (ggml_time_us() - t_start) / 1e6;

        lk.lock();

        it->writing = false;

        // the blocks are on disk even if the head failed: an erased entry removes them again
        for (const auto & [key, size] : written) {
            auto bit = blocks.find({ fingerprint, key });
            if (bit != blocks.end()) {
                bit->second.on_disk = true;
                bit->second.size    = size;
                bit->second.data.reset();
            }
        }

        if (ok) {
            it->on_disk = true;
            it->size    = fs::file_size(fs::u8path(path), ec);

            n_written += it->size;

            SRV_INF("disk prompt cache: wrote prompt with %zu tokens (%zu new KV block(s), %.3f MiB) in %.2f s, %.3f MiB on disk\n",
                    it->tokens.size(), written.size(), n_written / MiB, t_s, size_on_disk() / MiB);

            if (!it->cancel) {
                // keep the state if fetch() is waiting for it
                size_pending -= it->size_mem;
                it->pending.reset();
            }

            drop_superseded(*it);

            make_room(0, id, it->t_used);
        } else if (!it->cancel) {
            SRV_WRN("disk prompt cache: failed to write '%s', dropping prompt with %zu tokens\n", path.c_str(), it->tokens.size());
            erase_entry(it);
        }

        cv.notify_all();
    }
}

// ranged state: a head file and one file per KV block, native byte order
//   head:  u32 magic, u32 version, u64 fingerprint, u64 n + n x llama_token (server_tokens::serialize()),
//          u32 n_blocks + n_blocks x 2 u64 key, then the KV after the blocks (target, draft) and the partial state
//          (target, draft), each as u64 n + n bytes
//   block: u32 magic, u32 version, u64 fingerprint, 2 u64 key, u64 n_begin, u64 n_end, then the KV (target, draft)

bool server_prompt_cache_disk::write_head(const std::string & path, const server_prompt_cache_state & state, const std::atomic<bool> & cancel) const {
    disk_file f(path, "wb");
    if (!f.ok) {
        return false;
    }

    const std::vector<char> packed = state.prompt.tokens.serialize();
    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);

    f.write_val<uint32_t>(DISK_MAGIC_HEAD);
    f.write_val<uint32_t>(DISK_VERSION_KV);
    f.write_val<uint64_t>(fingerprint);
    f.write_val<uint64_t>(packed.size() / sizeof(llama_token));
    f.write(packed.data(), packed.size());

    f.write_val<uint32_t>((uint32_t) state.data.blocks.size());
    for (const auto & block : state.data.blocks) {
        f.write_val<uint64_t>(block->key.a);
        f.write_val<uint64_t>(block->key.b);
    }

    if (!write_blob(f, state.data.tail_tgt, cancel) ||
        !write_blob(f, state.data.tail_dft, cancel) ||
        !write_blob(f, state.data.part_tgt, cancel) ||
        !write_blob(f, state.data.part_dft, cancel)) {
        return false;
    }

    return f.sync();
}

bool server_prompt_cache_disk::read_head(const std::string & path, uint64_t fp, server_prompt_cache_state & out, std::vector<server_kv_key> & keys, bool header_only) const {
    std::error_code ec;
    const size_t file_size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }

    disk_file f(path, "rb");

    if (f.read_val<uint32_t>() != DISK_MAGIC_HEAD ||
        f.read_val<uint32_t>() != DISK_VERSION_KV ||
        f.read_val<uint64_t>() != fp              || !f.ok) {
        return false;
    }

    const uint64_t n_packed = f.read_val<uint64_t>();
    if (!f.ok || n_packed > (file_size - f.n) / sizeof(llama_token)) {
        return false;
    }

    llama_tokens packed(n_packed);
    f.read(packed.data(), n_packed * sizeof(llama_token));
    if (!f.ok) {
        return false;
    }

    try {
        out.prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: invalid tokens in '%s': %s\n", path.c_str(), err.what());
        return false;
    }

    const uint32_t n_blocks = f.read_val<uint32_t>();
    if (!f.ok || n_blocks > (file_size - f.n) / (2*sizeof(uint64_t))) {
        return false;
    }

    keys.resize(n_blocks);
    for (auto & key : keys) {
        key.a = f.read_val<uint64_t>();
        key.b = f.read_val<uint64_t>();
    }
    if (!f.ok) {
        return false;
    }

    out.data.ranged = true;

    if (header_only) {
        return true;
    }

    if (!read_blob(f, out.data.tail_tgt, file_size) ||
        !read_blob(f, out.data.tail_dft, file_size) ||
        !read_blob(f, out.data.part_tgt, file_size) ||
        !read_blob(f, out.data.part_dft, file_size)) {
        return false;
    }

    return f.ok && f.n == file_size;
}

bool server_prompt_cache_disk::write_block(const std::string & path, const server_kv_block & block, const std::atomic<bool> & cancel) const {
    disk_file f(path, "wb");
    if (!f.ok) {
        return false;
    }

    f.write_val<uint32_t>(DISK_MAGIC_BLOCK);
    f.write_val<uint32_t>(DISK_VERSION_KV);
    f.write_val<uint64_t>(fingerprint);
    f.write_val<uint64_t>(block.key.a);
    f.write_val<uint64_t>(block.key.b);
    f.write_val<uint64_t>(block.n_begin);
    f.write_val<uint64_t>(block.n_end);

    if (!write_blob(f, block.tgt, cancel) ||
        !write_blob(f, block.dft, cancel)) {
        return false;
    }

    return f.sync();
}

bool server_prompt_cache_disk::read_block(const std::string & path, uint64_t fp, server_kv_block & out, bool header_only) const {
    std::error_code ec;
    const size_t file_size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }

    disk_file f(path, "rb");

    if (f.read_val<uint32_t>() != DISK_MAGIC_BLOCK ||
        f.read_val<uint32_t>() != DISK_VERSION_KV  ||
        f.read_val<uint64_t>() != fp               || !f.ok) {
        return false;
    }

    out.key.a   = f.read_val<uint64_t>();
    out.key.b   = f.read_val<uint64_t>();
    out.n_begin = f.read_val<uint64_t>();
    out.n_end   = f.read_val<uint64_t>();

    if (!f.ok || out.n_end <= out.n_begin) {
        return false;
    }

    if (header_only) {
        return true;
    }

    if (!read_blob(f, out.tgt, file_size) ||
        !read_blob(f, out.dft, file_size)) {
        return false;
    }

    return f.ok && f.n == file_size;
}

// whole state: one file
// file layout, native byte order:
//   u32 magic, u32 version, u64 fingerprint
//   u64 n, n x llama_token: server_tokens::serialize()
//   u64 n, n bytes: target state
//   u64 n, n bytes: draft state
//   u32 n_checkpoints, then per checkpoint:
//     i64 n_tokens, i32 pos_min, i32 pos_max, and the target, draft and speculative data as u64 n + n bytes

bool server_prompt_cache_disk::write_file(const std::string & path, const server_prompt_cache_state & state, const std::atomic<bool> & cancel) const {
    disk_file f(path, "wb");
    if (!f.ok) {
        return false;
    }

    const std::vector<char> packed = state.prompt.tokens.serialize();
    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);

    f.write_val<uint32_t>(DISK_MAGIC);
    f.write_val<uint32_t>(DISK_VERSION);
    f.write_val<uint64_t>(fingerprint);
    f.write_val<uint64_t>(packed.size() / sizeof(llama_token));
    f.write(packed.data(), packed.size());

    if (!write_blob(f, state.data.main, cancel) ||
        !write_blob(f, state.data.drft, cancel)) {
        return false;
    }

    f.write_val<uint32_t>((uint32_t) state.prompt.checkpoints.size());

    for (const auto & cur : state.prompt.checkpoints) {
        f.write_val<int64_t>  (cur.n_tokens);
        f.write_val<llama_pos>(cur.pos_min);
        f.write_val<llama_pos>(cur.pos_max);

        if (!write_blob(f, cur.data_tgt,  cancel) ||
            !write_blob(f, cur.data_dft,  cancel) ||
            !write_blob(f, cur.data_spec, cancel)) {
            return false;
        }
    }

    return f.sync();
}

bool server_prompt_cache_disk::read_file(const std::string & path, server_prompt_cache_state & out, bool tokens_only) const {
    std::error_code ec;
    const size_t file_size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }

    disk_file f(path, "rb");

    if (f.read_val<uint32_t>() != DISK_MAGIC   ||
        f.read_val<uint32_t>() != DISK_VERSION ||
        f.read_val<uint64_t>() != fingerprint  || !f.ok) {
        return false;
    }

    const uint64_t n_packed = f.read_val<uint64_t>();
    if (!f.ok || n_packed > (file_size - f.n) / sizeof(llama_token)) {
        return false;
    }

    llama_tokens packed(n_packed);
    f.read(packed.data(), n_packed * sizeof(llama_token));
    if (!f.ok) {
        return false;
    }

    try {
        out.prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: invalid tokens in '%s': %s\n", path.c_str(), err.what());
        return false;
    }

    if (tokens_only) {
        return true;
    }

    if (!read_blob(f, out.data.main, file_size) ||
        !read_blob(f, out.data.drft, file_size)) {
        return false;
    }

    const uint32_t n_checkpoints = f.read_val<uint32_t>();

    out.prompt.checkpoints.clear();

    for (uint32_t i = 0; i < n_checkpoints && f.ok; ++i) {
        common_prompt_checkpoint cur;

        cur.id_task  = -1; // not created by a task: a checkpoint that fails to load falls back to reprocessing
        cur.n_tokens = f.read_val<int64_t>();
        cur.pos_min  = f.read_val<llama_pos>();
        cur.pos_max  = f.read_val<llama_pos>();

        if (!read_blob(f, cur.data_tgt,  file_size) ||
            !read_blob(f, cur.data_dft,  file_size) ||
            !read_blob(f, cur.data_spec, file_size) || cur.data_tgt.empty()) {
            return false;
        }

        out.prompt.checkpoints.push_back(std::move(cur));
    }

    return f.ok && f.n == file_size;
}

static bool file_fingerprint(uint64_t & h, const std::string & path) {
    std::error_code ec;
    const uint64_t size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }

    disk_file f(path, "rb");
    if (!f.ok) {
        return false;
    }

    h = fnv1a(h, &size, sizeof(size));

    // 16 samples of 4 KiB spread over the file tell apart models with the same size and metadata
    constexpr size_t n_samples   = 16;
    constexpr size_t sample_size = 4096;

    std::vector<uint8_t> buf(sample_size);

    for (size_t i = 0; i < n_samples; ++i) {
        const uint64_t offset = size > sample_size ? (size - sample_size) * i / (n_samples - 1) : 0;
        if (!f.seek(offset)) {
            return false;
        }

        const size_t n = fread(buf.data(), 1, sample_size, f.f);
        h = fnv1a(h, buf.data(), n);
    }

    return true;
}

uint64_t server_prompt_cache_disk::compute_fingerprint(const std::vector<std::string> & files, const std::string & config) {
    uint64_t h = fnv1a(FNV_OFFSET, &DISK_VERSION, sizeof(DISK_VERSION));

    for (const auto & path : files) {
        if (!file_fingerprint(h, path)) {
            SRV_WRN("disk prompt cache: cannot read '%s'\n", path.c_str());
            return 0;
        }
    }

    h = fnv1a(h, config.data(), config.size());

    return h;
}

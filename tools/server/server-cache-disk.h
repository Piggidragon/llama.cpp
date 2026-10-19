#pragma once

#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// disk level of the prompt cache (--cache-disk)
//
// prompt states evicted from the RAM level are written to the cache directory and are loaded back when a later
// request shares their prefix. the files are indexed again on startup, so the cache outlives the process.
// files are written by a background thread: an evicted state stays in memory until its files are complete, and a
// request that needs it in the meantime takes it back from the queue.
//
// a ranged state (see server_prompt_data) is written as a head file (tokens, block keys, the KV after the last
// block and the partial state) and one file per KV block. blocks are shared by all heads that hold the same
// tokens, so a conversation that grows writes only its new blocks. a state of a memory without ranges is written
// as one file.
//
// file names start with a fingerprint of the model files and of the settings that change the KV contents or the
// state format, so several models and configurations can share one directory. files of other fingerprints are
// never loaded, but they count toward the size limit: their heads are evicted by LRU like the own ones. their
// blocks are evicted with the last head that holds them, and only while no server of their fingerprint runs: it
// may hold them in memory. a server holds a lock file for its fingerprint, so only one server of a fingerprint
// uses the directory at a time.
struct server_prompt_cache_disk {
    using time_point = std::filesystem::file_time_type;

    // KV blocks that the RAM level holds, so that fetch() does not read them from disk; may return nullptr
    using block_source = std::function<server_kv_block_ptr(const server_kv_key & key)>;

    // exclusive lock on a file, released when destroyed or when the process exits
    struct file_lock {
        enum class result { locked, busy, failed };

        file_lock() = default;
        file_lock(file_lock && other) noexcept;
        file_lock & operator=(file_lock && other) noexcept;
        ~file_lock();

        // does not block, creates the file if needed
        result try_lock(const std::string & path);
        void   release();

    private:
        intptr_t handle = -1;
    };

    // the disk level for a fingerprint, nullptr if another server uses the directory with the same fingerprint
    // limit_size:    bytes in the directory, 0 = no limit
    // limit_pending: bytes of states waiting to be written, 0 = no limit; beyond it the oldest waiting state is dropped
    static std::unique_ptr<server_prompt_cache_disk> create(const std::string & dir, size_t limit_size, size_t limit_pending, uint64_t fingerprint, bool has_mtmd);

    // lock: the lock of the fingerprint, see create()
    server_prompt_cache_disk(const std::string & dir, size_t limit_size, size_t limit_pending, uint64_t fingerprint, bool has_mtmd, file_lock && lock);

    // writes the states still waiting before returning
    ~server_prompt_cache_disk();

    // take a state evicted from RAM and write it in the background, t_used is its last use for the LRU order
    // nothing is written if an entry already holds these tokens or a continuation of them
    // bounded = false skips the limit on waiting states (the caller only moves states that are already in memory)
    void put(server_prompt_cache_state && state, time_point t_used = time_point::clock::now(), bool bounded = true);

    // true if an entry (written or waiting) holds these tokens or a continuation of them
    bool contains(const server_tokens & tokens) const;

    // id of the entry that shares a longer prefix than lcp_best with tokens_new, -1 if none
    // updates lcp_best; entries with more than n_tokens_max tokens are skipped, and with whole_only the heads that
    // tokens_new does not hold whole (stored without checkpoints, a memory that rolls back only to checkpoints cannot use them)
    int64_t find(const server_tokens & tokens_new, size_t n_tokens_max, size_t & lcp_best, bool whole_only = false) const;

    // move the state of an entry into out: a waiting state is taken back, a written one is read from its files
    // a corrupt file is deleted
    bool fetch(int64_t id, server_prompt_cache_state & out, const block_source & source = nullptr);

    // the state fetched for id did not restore into the context. permanent: the failure cannot be for a lack of
    // free cells (each slot has its own KV stream), so the stored state is broken. a broken head is removed with
    // its blocks, so that the prompts that share them write them again; a whole state is skipped from then on.
    // a failure that may be for a lack of cells is counted, and the entry is skipped after the second one; its
    // files are kept, the blocks may be fine and shared by other heads.
    // a state that was taken back from the queue is queued again
    void restore_failed(int64_t id, server_prompt_cache_state && state, bool permanent);

    // block until all waiting states are written
    void flush();

    size_t n_entries() const;

    // bytes of the files in the directory
    size_t size() const;

    const std::string & get_dir() const { return dir; }
    uint64_t get_fingerprint()    const { return fingerprint; }

    // hash of the size and sampled content of the files and of the config string, 0 if a file cannot be read
    static uint64_t compute_fingerprint(const std::vector<std::string> & files, const std::string & config);

private:
    struct entry {
        int64_t     id;
        std::string path;

        uint64_t fp = 0; // fingerprint of the file

        server_tokens tokens;

        bool ranged = false;

        // a root prompt (server_prompt_cache_state::root), file kind 'r': the prompts that continue it do not remove it
        bool root = false;

        // keys of the KV blocks of a ranged state, in position order
        std::vector<server_kv_key> blocks;

        size_t size     = 0; // file size once written, state size while waiting
        size_t size_mem = 0; // bytes of the waiting state that are not on disk yet

        time_point t_used; // last use, kept as the file mtime

        bool on_disk = false;
        bool foreign = false; // file of another fingerprint: counts toward the limit, never loaded
        bool writing = false;
        bool reading = false;

        int n_fail = 0; // failed restores, the entry is skipped after 2

        // set by fetch() while the entry is being written, the writer then stops and keeps the state
        std::atomic<bool> cancel{false};

        // state waiting to be written
        std::unique_ptr<server_prompt_cache_state> pending;

        bool busy() const { return writing || reading; }
    };

    // a KV block file, shared by the heads of its fingerprint that list its key
    struct block_id {
        uint64_t      fp;
        server_kv_key key;

        bool operator==(const block_id & other) const {
            return fp == other.fp && key == other.key;
        }
    };

    struct block_id_hash {
        size_t operator()(const block_id & id) const {
            return server_kv_key_hash()(id.key) ^ id.fp;
        }
    };

    struct block {
        std::string path;

        size_t size = 0;
        int    refs = 0;

        bool on_disk = false;

        // the block data while it waits to be written
        server_kv_block_ptr data;
    };

    std::string dir;
    std::string prefix; // file name prefix for the current fingerprint

    file_lock lock_own;

    // locks of other fingerprints whose servers do not run, taken by refresh_foreign() and held until make_room()
    // has removed their files
    std::vector<file_lock> locks_foreign;

    size_t limit_size    = 0;
    size_t limit_pending = 0;

    uint64_t fingerprint = 0;
    bool     has_mtmd    = false;

    mutable std::mutex      mtx;
    std::condition_variable cv;

    std::list<entry>    entries;
    std::deque<int64_t> queue; // ids of the entries waiting to be written, oldest first

    std::unordered_map<block_id, block, block_id_hash> blocks;

    size_t  size_pending = 0;
    int64_t id_next      = 0;
    bool    stopping     = false;

    std::thread worker;

    void worker_loop();

    std::list<entry>::iterator find_entry(int64_t id);

    void scan();

    // index the files of other fingerprints again, with the references of their blocks from their heads on disk
    // the blocks of a fingerprint whose server runs are not referenced, they are never removed
    void refresh_foreign();

    static std::string lock_path(const std::string & dir, uint64_t fingerprint);

    std::string block_path(const server_kv_key & key) const;

    // must be called with mtx held
    size_t size_on_disk() const;
    bool   make_room(size_t size_new, int64_t id_keep, time_point t_new);
    void   erase_entry(std::list<entry>::iterator it);
    void   drop_superseded(const entry & cur);
    void   release_blocks(const entry & e);

    bool write_file (const std::string & path, const server_prompt_cache_state & state, const std::atomic<bool> & cancel) const;
    bool read_file  (const std::string & path, server_prompt_cache_state & out, bool tokens_only) const;
    bool write_head (const std::string & path, const server_prompt_cache_state & state, const std::atomic<bool> & cancel) const;
    bool read_head  (const std::string & path, uint64_t fp, server_prompt_cache_state & out, std::vector<server_kv_key> & keys, bool header_only) const;
    bool write_block(const std::string & path, const server_kv_block & block, const std::atomic<bool> & cancel) const;
    bool read_block (const std::string & path, uint64_t fp, server_kv_block & out, bool header_only) const;
};

import os
import glob
import shutil
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

cache_dir: str = ""


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def make_server():
    srv = ServerPreset.tinyllama2()
    srv.n_slots = 1
    srv.n_predict = 4
    srv.temperature = 0.0
    srv.cache_ram = 1
    srv.debug = True
    srv.cache_disk = cache_dir
    fd, srv.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    return srv


@pytest.fixture(autouse=True)
def create_server():
    global server, cache_dir
    cache_dir = tempfile.mkdtemp(prefix='prompt-cache-disk-')
    server = make_server()
    yield
    server.stop()
    shutil.rmtree(cache_dir, ignore_errors=True)


def cache_files():
    return glob.glob(os.path.join(cache_dir, "pc-*.bin"))


def head_files():
    return [f for f in cache_files() if os.path.basename(f).split("-")[2].startswith("h")]


def block_files():
    return [f for f in cache_files() if os.path.basename(f).split("-")[2].startswith("k")]


def head_blocks(path: str) -> list[int]:
    # head: u32 magic, u32 version, u64 fingerprint, u64 n + n packed tokens, u32 n_blocks + n_blocks 16-byte keys
    with open(path, "rb") as f:
        f.seek(16)
        n_packed = int.from_bytes(f.read(8), "little")
        f.seek(24 + 4 * n_packed)
        n_blocks = int.from_bytes(f.read(4), "little")
        return [int.from_bytes(f.read(16), "little") for _ in range(n_blocks)]


# distinct prompts of about 150 tokens each, so that a few of them fill the 1 MiB RAM level
def make_prompt(i: int) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett"]
    return f"Story number {i}: " + " ".join(words[(i + k) % len(words)] + str(k * (i + 1)) for k in range(60))


def complete(prompt: str):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body


@pytest.mark.parametrize("block_size", [None, 0])
def test_restore_after_restart(block_size):
    global server
    server.cache_block_size = block_size
    server.start()

    prompt = make_prompt(0)
    first = complete(prompt)
    assert first["timings"]["prompt_n"] > 100

    # the slot is written to disk on exit
    server.stop()
    assert len(cache_files()) == 1

    server = make_server()
    server.start()
    log = LogReader(server.log_path)
    assert "disk prompt cache: 1 prompt(s)" in log.drain()

    res = complete(prompt)
    assert "disk prompt cache: read prompt" in log.drain()
    assert res["timings"]["cache_n"] > 0
    assert res["timings"]["prompt_n"] < first["timings"]["prompt_n"]
    assert res["content"] == first["content"]


@pytest.mark.parametrize("block_size", [None, 0])
def test_spill_on_eviction(block_size):
    global server
    server.cache_block_size = block_size
    server.start()
    log = LogReader(server.log_path)

    first = complete(make_prompt(0))

    # each new prompt moves the previous one into the RAM level, which evicts to disk once full
    for i in range(1, 12):
        complete(make_prompt(i))

    content = log.drain()
    assert "removing oldest entry" in content
    assert "disk prompt cache: queued prompt" in content or "disk prompt cache: wrote prompt" in content

    # the first prompt is gone from RAM, it comes back from disk (or from the write queue)
    res = complete(make_prompt(0))
    content = log.drain()
    assert "disk prompt cache: read prompt" in content or "disk prompt cache: taking back prompt" in content
    assert res["timings"]["cache_n"] > 0
    assert res["timings"]["prompt_n"] < first["timings"]["prompt_n"]
    assert res["content"] == first["content"]


def test_other_model_files_ignored():
    global server
    server.start()
    complete(make_prompt(0))
    server.stop()

    files = cache_files()
    assert len(files) == 1

    # a file of another fingerprint is neither indexed nor deleted
    other = os.path.join(cache_dir, "pc-0000000000000000-0000000000000000.bin")
    shutil.copy(files[0], other)

    server = make_server()
    server.start()
    log = LogReader(server.log_path)
    assert "disk prompt cache: 1 prompt(s)" in log.drain()
    assert os.path.exists(other)


def n_tokens_in_file(path: str) -> int:
    # header: u32 magic, u32 version, u64 fingerprint, u64 number of packed tokens
    with open(path, "rb") as f:
        f.seek(16)
        return int.from_bytes(f.read(8), "little")


def test_lru_order_kept_on_exit():
    global server
    server.start()

    # A, then B, then C (C stays in the slot, A and B go to the RAM level)
    for i, n in enumerate([20, 40, 60]):
        complete(make_prompt(i)[: 12 * n])
    server.stop()

    files = sorted(cache_files(), key=os.path.getmtime)
    assert len(files) == 3
    # the newest state (the slot) gets the newest mtime, so it is evicted last
    sizes = [n_tokens_in_file(f) for f in files]
    assert sizes == sorted(sizes)


def test_disk_size_zero_disables():
    global server
    server.cache_disk_size = 0
    server.start()
    log = LogReader(server.log_path)
    assert "disk prompt cache disabled by --cache-disk-size 0" in log.drain()
    complete(make_prompt(0))
    server.stop()
    assert len(cache_files()) == 0


def test_other_fingerprints_count_toward_limit():
    global server
    # an old file of another fingerprint, larger than the limit
    other = os.path.join(cache_dir, "pc-0000000000000000-0000000000000000.bin")
    with open(other, "wb") as f:
        f.write(b"\0" * (3 * 1024 * 1024))
    os.utime(other, (1, 1))

    server.cache_disk_size = 2
    server.start()
    log = LogReader(server.log_path)
    assert "removing least recently used file of another fingerprint" in log.drain()
    assert not os.path.exists(other)


@pytest.mark.parametrize("block_size", [None, 0])
def test_corrupt_file_removed(block_size):
    global server
    server.cache_block_size = block_size
    server.start()
    first = complete(make_prompt(0))
    server.stop()

    files = cache_files()
    assert len(files) == 1

    # cut the state, the header and tokens stay valid so the file is indexed
    size = os.path.getsize(files[0])
    with open(files[0], "r+b") as f:
        f.truncate(size // 2)

    server = make_server()
    server.start()
    log = LogReader(server.log_path)

    res = complete(make_prompt(0))
    assert "is corrupt, removing it" in log.drain()
    assert res["content"] == first["content"]
    assert not os.path.exists(files[0])


def test_cached_prompt_beats_slot():
    global server
    server.cache_ram = 100
    server.start()
    log = LogReader(server.log_path)

    def n_tokens(text: str) -> int:
        res = server.make_request("POST", "/tokenize", data={"content": text})
        return len(res.body["tokens"])

    system = "You are a helpful assistant who answers questions about stories, places and people. " * 2
    story = " ".join(f"word{k}" for k in range(66))

    # the slot is picked by similarity (> 0.1), and the cache gains more than 256 tokens over it
    n_system = n_tokens(system)
    n_story  = n_tokens(system + story) - n_system
    assert n_system > 0.1 * (n_system + n_story + 8)
    assert n_story > 300
    assert n_system + n_story + 16 < 512

    # conversation A fills the slot
    a = complete(system + story)

    # conversation B shares only the system prompt and keeps most of the slot, A moves to the RAM cache
    complete(system + "Hello there")

    # A continues: the slot shares only the system prompt, the cache holds all of A
    res = complete(system + story + " The end.")
    assert "prompt cache shares a longer prefix with the task" in log.drain()
    assert res["timings"]["cache_n"] >= a["timings"]["prompt_n"] - 8
    assert res["timings"]["prompt_n"] < 32


def test_only_new_blocks_written():
    global server
    server.cache_block_size = 64
    server.n_ctx = 2048
    server.start()

    prompt = make_prompt(0)[:900]
    first = complete(prompt)
    server.stop()

    heads = head_files()
    assert len(heads) == 1
    keys_before = set(head_blocks(heads[0]))
    blocks_before = {f: (os.stat(f).st_ino, os.stat(f).st_mtime_ns) for f in block_files()}
    assert len(blocks_before) == len(keys_before) > 0

    # the conversation continues after a restart: it is restored from disk and grows by more than a block
    server = make_server()
    server.cache_block_size = 64
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)
    res = complete(prompt + first["content"] + " " + make_prompt(5)[:400])
    assert res["timings"]["cache_n"] >= first["timings"]["prompt_n"] - 8
    server.stop()

    # the new head lists the old blocks, which are not written again; only the new blocks are
    newest = max(head_files(), key=os.path.getmtime)
    keys = set(head_blocks(newest))
    assert keys_before <= keys
    for f, sig in blocks_before.items():
        assert os.path.exists(f)
        assert (os.stat(f).st_ino, os.stat(f).st_mtime_ns) == sig

    content = log.drain()
    n_new = sum(int(l.split("(")[1].split(" new KV")[0]) for l in content.splitlines() if "disk prompt cache: wrote prompt" in l)
    assert n_new == len(keys - keys_before) > 0
    assert len(block_files()) == len(keys)


def test_shared_prefix_blocks_stored_once():
    global server
    server.cache_block_size = 64
    server.n_slots = 2
    server.n_ctx = 1024
    server.start()

    # two conversations in two slots that share a long prefix, both written on exit
    shared = make_prompt(1)[:1400]
    for i, ending in enumerate([" The first ending is about alpha and bravo.", " The second ending is about charlie and delta, which is longer."]):
        res = server.make_request("POST", "/completion", data={"prompt": shared + ending, "cache_prompt": True, "id_slot": i})
        assert res.status_code == 200
    server.stop()

    heads = head_files()
    assert len(heads) == 2
    keys_a, keys_b = (set(head_blocks(h)) for h in heads)
    assert len(keys_a & keys_b) > 0
    assert len(block_files()) == len(keys_a | keys_b)


def tokens_of(seed: int, n: int) -> list[int]:
    return [10 + ((seed * 13 + k * 7) % 480) for k in range(n)]


def complete_tokens(tokens: list[int]):
    res = server.make_request("POST", "/completion", data={"prompt": tokens, "cache_prompt": True})
    assert res.status_code == 200
    return res.body


def run_once(opts: dict, prompt: list[int]):
    global server
    server = make_server()
    server.n_ctx = 2048
    for k, v in opts.items():
        setattr(server, k, v)
    server.start()
    log = LogReader(server.log_path)
    res = complete_tokens(prompt)
    server.stop()
    return res, log.drain()


def test_attention_layout_change_gets_own_files():
    # flash attention changes the layout of the V cache: the old files must not be used or shared
    prompt = tokens_of(0, 300)

    res, _ = run_once({"fa": "off", "cache_block_size": 32}, prompt)
    assert res["timings"]["cache_n"] == 0

    res, log = run_once({"fa": "on", "cache_block_size": 32}, prompt)
    assert "file(s) of other fingerprints" in log and " 0 prompt(s)" in log
    assert "did not restore" not in log and "is corrupt" not in log
    assert res["timings"]["cache_n"] == 0

    # the new layout restores its own files
    res, log = run_once({"fa": "on", "cache_block_size": 32}, prompt)
    assert "did not restore" not in log
    assert res["timings"]["cache_n"] > 250


def test_block_size_change_keeps_cache():
    # blocks of another size have other keys: the old prompt still restores, the new one writes its own blocks
    prompt = tokens_of(0, 300)

    run_once({"cache_block_size": 32}, prompt)
    n_before = len(block_files())

    res, log = run_once({"cache_block_size": 64}, prompt + tokens_of(1, 100))
    assert "disk prompt cache: read prompt" in log
    assert "is corrupt" not in log and "did not restore" not in log
    assert res["timings"]["cache_n"] > 250
    assert len(block_files()) > n_before

    res, log = run_once({"cache_block_size": 64}, prompt + tokens_of(1, 100))
    assert "is corrupt" not in log and "did not restore" not in log
    assert res["timings"]["cache_n"] > 350


def test_large_prompt_with_shared_blocks_skipped():
    # a prompt larger than --cache-ram is not cached, also when it shares blocks with cached prompts
    global server
    server.cache_block_size = 32
    server.n_ctx = 4096
    server.n_predict = 2
    server.start()
    log = LogReader(server.log_path)

    system = tokens_of(0, 400)
    x = system + tokens_of(1, 500)
    complete_tokens(tokens_of(2, 300))
    complete_tokens(x)
    complete_tokens(tokens_of(4, 50))
    complete_tokens(system + tokens_of(3, 1300))
    complete_tokens(tokens_of(5, 50))
    assert "exceeds cache size limit" in log.drain()

    # x is still cached in RAM
    res = complete_tokens(x)
    assert res["timings"]["cache_n"] > 800


def test_corrupt_block_written_again():
    global server
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.start()
    prompt = tokens_of(0, 300)
    first = complete_tokens(prompt)
    server.stop()

    blocks = sorted(block_files())
    assert len(blocks) > 2
    with open(blocks[1], "r+b") as f:
        f.truncate(os.path.getsize(blocks[1]) // 2)

    server = make_server()
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)
    res = complete_tokens(prompt)
    assert "is corrupt, removing it" in log.drain()
    assert res["content"] == first["content"]
    server.stop()

    # the next write of the prompt writes the block again
    server = make_server()
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)
    res = complete_tokens(prompt)
    assert "disk prompt cache: read prompt" in log.drain()
    assert res["timings"]["cache_n"] > 250


def test_other_fingerprint_evicted_with_its_blocks():
    # the heads of another fingerprint are evicted with the blocks that only they hold
    global server
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.ctk = "f32"
    server.start()
    complete_tokens(tokens_of(0, 300))
    server.stop()
    other = set(cache_files())
    assert len(other) > 2

    # both prompts together are more than 1 MiB, the new one alone fits
    server = make_server()
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.cache_disk_size = 1
    server.start()
    complete_tokens(tokens_of(1, 1200))
    server.stop()
    assert len(head_files()) == 1

    # the other model's prompt is gone completely, no orphan blocks are left
    assert not (other & set(cache_files()))


@pytest.mark.skipif(not os.environ.get("LLAMA_TEST_HYBRID_MODEL"), reason="needs LLAMA_TEST_HYBRID_MODEL, a small hybrid/recurrent GGUF")
def test_hybrid_restore_at_checkpoint():
    # a hybrid model stores a prompt at its newest checkpoint; the restored state continues like a fresh one
    global server

    def run(cache):
        global server
        server = make_server()
        server.model_hf_repo = None
        server.model_file = os.environ["LLAMA_TEST_HYBRID_MODEL"]
        server.n_ctx = 4096
        server.cache_block_size = 256
        server.n_predict = 16
        if not cache:
            server.cache_disk = None
        server.start()
        log = LogReader(server.log_path)
        prompt = " ".join(f"item{(k * 31) % 1000}" for k in range(500))
        first = complete(prompt)
        res = complete(prompt + first["content"] + " And then the story goes on.")
        server.stop()
        return first, res, log.drain()

    first, res, _ = run(cache=True)

    # restart: the conversation comes back from disk at the checkpoint
    server = make_server()
    server.model_hf_repo = None
    server.model_file = os.environ["LLAMA_TEST_HYBRID_MODEL"]
    server.n_ctx = 4096
    server.cache_block_size = 256
    server.n_predict = 16
    server.start()
    log = LogReader(server.log_path)
    prompt = " ".join(f"item{(k * 31) % 1000}" for k in range(500))
    cont = prompt + first["content"] + " And then the story goes on." + res["content"] + " The end."
    restored = complete(cont)
    content = log.drain()
    server.stop()
    assert "disk prompt cache: read prompt" in content
    assert "forcing full prompt re-processing" not in content
    assert restored["timings"]["cache_n"] > 1000

    # the same request on a server without cache gives the same text
    server = make_server()
    server.model_hf_repo = None
    server.model_file = os.environ["LLAMA_TEST_HYBRID_MODEL"]
    server.n_ctx = 4096
    server.n_predict = 16
    server.cache_disk = None
    server.start()
    fresh = complete(cont)
    server.stop()
    assert restored["content"] == fresh["content"]


def test_shared_directory_keeps_blocks_of_other_server():
    # a server that evicts another fingerprint's head keeps the blocks that newer heads of it hold
    global server

    def other_server():
        srv = make_server()
        srv.cache_block_size = 32
        srv.n_ctx = 2048
        srv.ctk = "f32"
        srv.server_port = server.server_port + 1
        return srv

    system = tokens_of(0, 256)

    # the other model writes C1, then this server starts, then the other model writes C2 with the same system prompt
    b = other_server()
    b.start()
    res = b.make_request("POST", "/completion", data={"prompt": system + tokens_of(1, 64), "cache_prompt": True})
    assert res.status_code == 200
    b.stop()

    server.cache_block_size = 32
    server.n_ctx = 2048
    server.cache_disk_size = 1
    server.start()

    b = other_server()
    b.start()
    c2 = system + tokens_of(2, 64)
    res = b.make_request("POST", "/completion", data={"prompt": c2, "cache_prompt": True})
    assert res.status_code == 200
    b.stop()

    # this server needs room, C1 with its own blocks is enough
    complete_tokens(tokens_of(3, 1050))
    server.stop()

    b = other_server()
    b.start()
    log = LogReader(b.log_path)
    res = b.make_request("POST", "/completion", data={"prompt": c2 + tokens_of(4, 20), "cache_prompt": True})
    b.stop()
    assert "is missing" not in log.drain()
    assert res.body["timings"]["cache_n"] > 300


def test_broken_head_written_again():
    # a block whose state does not load (another format) is removed with its head and written again
    # without unified KV a failed restore cannot be for a lack of cells, so the first failure counts
    global server
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.start()
    prompt = tokens_of(0, 300)
    complete_tokens(prompt)
    server.stop()

    # break the magic of the state range inside each block, the file itself stays valid
    for f in block_files():
        with open(f, "r+b") as fh:
            fh.seek(56)
            fh.write(b"\0\0\0\0")

    server = make_server()
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)
    res = complete_tokens(prompt)
    server.stop()
    content = log.drain()
    assert "did not restore" in content and "removing it and its KV blocks" in content
    assert res["timings"]["cache_n"] == 0

    # the run wrote the prompt again on exit, with new blocks
    server = make_server()
    server.cache_block_size = 32
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)
    res = complete_tokens(prompt)
    server.stop()
    assert "did not restore" not in log.drain()
    assert res["timings"]["cache_n"] > 250


@pytest.mark.skipif(not os.environ.get("LLAMA_TEST_HYBRID_MODEL"), reason="needs LLAMA_TEST_HYBRID_MODEL, a small hybrid/recurrent GGUF")
def test_hybrid_head_not_read_for_earlier_divergence():
    # a head is stored without checkpoints: a prompt that diverges before its end cannot use it, so it is not read
    global server

    def hybrid_server():
        srv = make_server()
        srv.model_hf_repo = None
        srv.model_file = os.environ["LLAMA_TEST_HYBRID_MODEL"]
        srv.n_ctx = 4096
        srv.cache_block_size = 256
        srv.n_predict = 4
        return srv

    prompt = " ".join(f"item{(k * 31) % 1000}" for k in range(500))
    server = hybrid_server()
    server.start()
    complete(prompt + " What is the last item?")
    server.stop()

    server = hybrid_server()
    server.start()
    log = LogReader(server.log_path)
    complete(prompt + " Which item comes first?")
    server.stop()
    content = log.drain()
    assert "disk prompt cache: read prompt" not in content
    assert "no context checkpoint before the common prefix" not in content

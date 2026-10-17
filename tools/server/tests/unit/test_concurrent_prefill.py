import os
import tempfile
import time
import pytest
from utils import *

server = ServerPreset.tinyllama2()

LONG_PROMPT = " ".join(f"word{(k * 37) % 997}" for k in range(150))
SHORT_PROMPT = "Write a short story about a cat."


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = 4096
    server.temperature = 0.0
    yield
    server.stop()


def start(concurrent: bool, model_file: str | None = None):
    global server
    if model_file:
        server.model_hf_repo = None
        server.model_file = model_file
    server.concurrent_prefill = concurrent
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    server.start()


def read_log() -> str:
    with open(server.log_path, errors="replace") as f:
        log = f.read()
    return log


def gen_and_long(long_prompt: str, n_long: int = 8):
    gen = lambda: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT, "n_predict": 1024, "ignore_eos": True, "id_slot": 0, "cache_prompt": False})
    def long():
        time.sleep(0.05) # arrives while slot 0 generates
        return server.make_request("POST", "/completion", data={"prompt": long_prompt, "n_predict": n_long, "id_slot": 1, "cache_prompt": False})
    res_gen, res_long = parallel_function_calls([(gen, ()), (long, ())])
    assert res_gen.status_code == 200 and res_long.status_code == 200
    return res_gen.body["content"], res_long.body["content"]


def run(concurrent: bool, model_file: str | None = None, long_prompt: str = LONG_PROMPT):
    start(concurrent, model_file)
    gen, long = gen_and_long(long_prompt)
    server.stop()
    log = read_log()
    os.remove(server.log_path)
    return gen, long, log


def test_concurrent_prefill_same_text():
    # the middle of a long prompt goes through the second context while slot 0 generates; both texts stay the same
    gen_ref, long_ref, log_ref = run(False)
    assert "concurrent prefill:" not in log_ref

    gen, long, log = run(True)
    assert "concurrent prefill: a prompt is computed in a second context" in log
    assert "prompt tokens in the second context" in log
    assert gen == gen_ref
    assert long == long_ref


def test_concurrent_prefill_cancel():
    # a request whose prompt is in the second context is cancelled; the slot then serves another request correctly
    start(True)
    gen = lambda: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT, "n_predict": 1024, "ignore_eos": True, "id_slot": 0, "cache_prompt": False})
    def long_cancelled():
        time.sleep(0.05)
        try:
            server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT * 2, "n_predict": 8, "id_slot": 1, "cache_prompt": False}, timeout=0.3)
        except Exception:
            pass # the client goes away, the server cancels the task
        return True
    parallel_function_calls([(gen, ()), (long_cancelled, ())])
    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "n_predict": 8, "id_slot": 1, "cache_prompt": False})
    assert res.status_code == 200
    after = res.body["content"]
    server.stop()
    assert "prompt tokens in the second context" in read_log()

    start(False)
    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "n_predict": 8, "id_slot": 1, "cache_prompt": False})
    assert res.status_code == 200
    assert after == res.body["content"]


@pytest.mark.skipif(not os.environ.get("LLAMA_TEST_HYBRID_MODEL"), reason="needs LLAMA_TEST_HYBRID_MODEL, a small hybrid/recurrent GGUF")
def test_concurrent_prefill_hybrid():
    # a hybrid model: the recurrent state of the prompt is computed in the second context, the checkpoints of the end
    # of the prompt in the first one; the text is the one without the second context
    hybrid = os.environ["LLAMA_TEST_HYBRID_MODEL"]
    prompt = " ".join(f"item{(k * 31) % 1000}" for k in range(450))
    gen_ref, long_ref, _ = run(False, hybrid, prompt)
    gen, long, log = run(True, hybrid, prompt)
    assert "prompt tokens in the second context" in log
    assert long == long_ref


def test_concurrent_prefill_single_slot():
    server.n_slots = 1
    start(True)
    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "n_predict": 8})
    assert res.status_code == 200
    server.stop()
    assert "--concurrent-prefill needs --parallel >= 2" in read_log()


@pytest.mark.skipif(not os.environ.get("LLAMA_TEST_HYBRID_MODEL"), reason="needs LLAMA_TEST_HYBRID_MODEL, a small hybrid/recurrent GGUF")
def test_concurrent_prefill_hybrid_three_slots():
    # the recurrent cell of the prompt's sequence (slot 0) lies between the cells of slots 1 and 2, which generate in
    # one ubatch of the target while the prompt goes through the second context: the target's ubatch must not reach
    # that cell (see llama_memory_recurrent::find_slot()). The prompt's text is the one without the flag.
    hybrid = os.environ["LLAMA_TEST_HYBRID_MODEL"]
    prompt = " ".join(f"item{(k * 31) % 1000}" for k in range(450))
    gen = lambda id_slot: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT + f" {id_slot}", "n_predict": 400, "ignore_eos": True, "id_slot": id_slot, "cache_prompt": False})
    short = lambda: server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 4, "id_slot": 0, "cache_prompt": False})
    long = lambda: server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 8, "id_slot": 0, "cache_prompt": False})

    def run3(concurrent: bool):
        global server
        server.n_slots = 3
        server.n_ctx = 3 * 2048
        start(concurrent, hybrid)
        # slot 1 generates in the first cell, slot 0 takes the second one, slot 2 generates in the third one
        def slot0():
            time.sleep(0.2)
            assert short().status_code == 200
            time.sleep(0.2) # slot 2 starts meanwhile
            return long()
        def slot2():
            time.sleep(0.3)
            return gen(2)
        res = parallel_function_calls([(gen, (1,)), (slot2, ()), (slot0, ())])
        server.stop()
        log = read_log()
        os.remove(server.log_path)
        assert all(r.status_code == 200 for r in res)
        return res[2].body["content"], log

    ref, _ = run3(False)
    out, log = run3(True)
    assert "prompt tokens in the second context" in log
    assert out == ref


def test_concurrent_prefill_cache_reuse():
    # the prompt of slot 1 reuses shifted chunks of its previous prompt (a KV shift for the next decode) while slot 0
    # generates: the shift is applied when the second context does not compute, the text is the one without the flag
    chunk_a = " ".join(f"alpha{k}" for k in range(40))
    chunk_b = " ".join(f"beta{k}" for k in range(40))
    # the second prompt drops a part of the first one: the chunks after it move to the left (a KV shift)
    first  = "Intro. " + " ".join(f"gamma{k}" for k in range(30)) + " " + chunk_a + " " + chunk_b
    second = "Intro. " + chunk_a + " " + chunk_b + " " + " ".join(f"delta{k}" for k in range(60))

    def run_reuse(concurrent: bool):
        global server
        server.cache_reuse = 8
        start(concurrent)
        res = server.make_request("POST", "/completion", data={"prompt": first, "n_predict": 4, "id_slot": 1, "cache_prompt": True})
        assert res.status_code == 200
        gen = lambda: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT, "n_predict": 1024, "ignore_eos": True, "id_slot": 0, "cache_prompt": False})
        def reuse():
            time.sleep(0.05)
            return server.make_request("POST", "/completion", data={"prompt": second, "n_predict": 8, "id_slot": 1, "cache_prompt": True})
        res_gen, res_reuse = parallel_function_calls([(gen, ()), (reuse, ())])
        server.stop()
        log = read_log()
        os.remove(server.log_path)
        assert res_gen.status_code == 200 and res_reuse.status_code == 200
        return res_reuse.body["content"], res_reuse.body["timings"]["cache_n"], log

    ref, _, _ = run_reuse(False)
    out, cache_n, log = run_reuse(True)
    # far more than the common prefix "Intro. " came from the cache: the shifted chunks were reused
    assert cache_n > 40
    assert "prompt tokens in the second context" in log
    assert out == ref


def test_concurrent_prefill_fitted_context():
    # without -c the target's context size comes from the fit, the second context takes it from the target
    server.n_ctx = None
    start(True)
    prompt = " ".join(f"w{k}" for k in range(60))
    gen, long = gen_and_long(prompt)
    server.stop()
    log = read_log()
    assert "failed to create the second context" not in log
    assert "prompt tokens in the second context" in log

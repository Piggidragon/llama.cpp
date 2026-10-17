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

import os
import tempfile
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


def run(budget: int | None):
    global server
    server.prefill_budget = budget
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    server.start()
    gen = lambda: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT, "n_predict": 1024, "ignore_eos": True, "id_slot": 0, "cache_prompt": False})
    long = lambda: server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "n_predict": 8, "id_slot": 1, "cache_prompt": False})
    # the long prompt arrives while slot 0 generates
    def long_later():
        import time
        time.sleep(0.05)
        return long()
    res_gen, res_long = parallel_function_calls([(gen, ()), (long_later, ())])
    server.stop()
    with open(server.log_path, errors="replace") as f:
        log = f.read()
    os.remove(server.log_path)
    assert res_gen.status_code == 200 and res_long.status_code == 200
    return res_gen.body["content"], res_long.body["content"], log


def test_prefill_budget_same_text():
    # a prompt processed in budget-sized steps next to a generating slot gives the same text as without a budget
    gen_ref, long_ref, log_ref = run(None)
    assert "prefill budget:" not in log_ref

    gen, long, log = run(16)
    assert "prefill budget: 16 prompt tokens in this step" in log
    assert gen == gen_ref
    assert long == long_ref


def test_prefill_budget_invalid():
    server.prefill_budget = -1
    with pytest.raises(Exception):
        server.start(timeout_seconds=15)

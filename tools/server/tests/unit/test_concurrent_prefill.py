import base64
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
    # slots 1 and 2 generate in one ubatch of the target while the prompt of slot 0 goes through the second context,
    # after the slots were used in the order 1, 0, 2: each sequence keeps the recurrent cell of its index, so the
    # target's ubatch does not reach the cell of slot 0 (see llama_memory_recurrent::find_slot()); without that the
    # memory is not independent and the second context is not used. The prompt's text is the one without the flag.
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


def test_concurrent_prefill_two_prompts():
    # three slots: slot 0 generates while slots 1 and 2 bring long prompts; the second prompt waits for the second
    # context instead of going into the target's batch, both go through it, both texts as without the flag
    def run_two(concurrent: bool):
        global server
        server.n_slots = 3
        server.n_ctx = 3 * 2048
        start(concurrent)
        gen = lambda: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT, "n_predict": 2048, "ignore_eos": True, "id_slot": 0, "cache_prompt": False})
        def long(id_slot, n):
            time.sleep(0.05)
            return server.make_request("POST", "/completion", data={"prompt": " ".join(f"tok{(k * 29 + id_slot) % 977}" for k in range(n)), "n_predict": 8, "id_slot": id_slot, "cache_prompt": False})
        res = parallel_function_calls([(gen, ()), (long, (1, 140)), (long, (2, 120))])
        server.stop()
        log = read_log()
        os.remove(server.log_path)
        assert all(r.status_code == 200 for r in res)
        return [res[1].body["content"], res[2].body["content"]], log

    ref, _ = run_two(False)
    out, log = run_two(True)
    assert "id  1 | task" in log and "id  2 | task" in log
    bg = [l for l in log.splitlines() if "prompt tokens in the second context" in l]
    assert any("id  1 |" in l for l in bg) and any("id  2 |" in l for l in bg)
    assert out == ref


@pytest.mark.skipif(not os.environ.get("LLAMA_TEST_QWEN4EXP_MODEL"), reason="needs LLAMA_TEST_QWEN4EXP_MODEL, e.g. from test-llama-archs -a qwen4exp -o DIR")
def test_concurrent_prefill_qwen4exp():
    # the indexer memory of qwen4exp: a ubatch writes only the pooled keys of its own sequences, so its prompt can go
    # through the second context; token prompts (the test model has a synthetic vocabulary), outputs as alone
    model = os.environ["LLAMA_TEST_QWEN4EXP_MODEL"]
    toks = lambda seed, n: [1 + ((seed * 13 + k * 7) % 120) for k in range(n)]

    def run_q(concurrent: bool, alone: bool):
        global server
        server.n_slots = 3
        server.n_ctx = 3 * 256
        server.n_batch = 32
        server.n_ubatch = 16
        start(concurrent, model)
        gen = lambda id_slot: server.make_request("POST", "/completion", data={"prompt": toks(id_slot, 20), "n_predict": 220, "ignore_eos": True, "id_slot": id_slot, "cache_prompt": False})
        def prompts():
            if not alone:
                time.sleep(0.15)
            r1 = server.make_request("POST", "/completion", data={"prompt": toks(7, 200), "n_predict": 16, "id_slot": 0, "cache_prompt": True, "return_tokens": True})
            r2 = server.make_request("POST", "/completion", data={"prompt": toks(7, 150) + toks(9, 80), "n_predict": 16, "id_slot": 0, "cache_prompt": True, "return_tokens": True})
            return [r1.body["tokens"], r2.body["tokens"]]
        if alone:
            out = prompts()
        else:
            out = parallel_function_calls([(gen, (1,)), (gen, (2,)), (prompts, ())])[2]
        server.stop()
        log = read_log()
        os.remove(server.log_path)
        return out, log

    ref, _ = run_q(False, alone=True)
    out, log = run_q(True, alone=False)
    assert "it is not used" not in log
    assert "prompt tokens in the second context" in log
    assert out == ref


def test_concurrent_prefill_mmproj():
    # with an mmproj loaded every chat prompt counts as multimodal, also a text-only one: its text goes through the
    # second context; an image in the middle of a prompt is decoded in the first one, the text after it goes through
    # the second context again; the texts are the ones without the flag
    with open(os.path.join(os.path.dirname(__file__), "../../../mtmd/test-1.jpeg"), "rb") as f:
        image = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    text_a = " ".join(f"word{(k * 37) % 997}" for k in range(300))
    text_b = " ".join(f"item{(k * 31) % 1000}" for k in range(300))
    text_only  = [{"role": "user", "content": text_a}]
    with_image = [{"role": "user", "content": [
        {"type": "text", "text": text_a},
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": text_b},
    ]}]

    def chat_and_gen(messages):
        gen = lambda: server.make_request("POST", "/completion", data={"prompt": SHORT_PROMPT, "n_predict": 1024, "ignore_eos": True, "id_slot": 0, "cache_prompt": False})
        def chat():
            time.sleep(0.05) # arrives while slot 0 generates
            return server.make_request("POST", "/v1/chat/completions", data={"messages": messages, "max_tokens": 8, "id_slot": 1, "cache_prompt": False})
        res_gen, res_chat = parallel_function_calls([(gen, ()), (chat, ())])
        assert res_gen.status_code == 200 and res_chat.status_code == 200
        return res_chat.body["choices"][0]["message"]["content"], read_log()

    def run_mm(concurrent: bool):
        global server
        server = ServerPreset.tinygemma3()
        server.n_slots = 2
        server.n_ctx = 8192
        server.temperature = 0.0
        start(concurrent)
        text, log_text = chat_and_gen(text_only)
        image_text, log_all = chat_and_gen(with_image)
        server.stop()
        os.remove(server.log_path)
        return text, image_text, log_text, log_all[len(log_text):]

    bg_line = "prompt tokens in the second context"
    text_ref, image_ref, _, _ = run_mm(False)
    text, image_text, log_text, log_image = run_mm(True)
    assert bg_line in log_text
    # the text after the image goes through the second context too
    i_image = log_image.find("image decoded")
    assert i_image >= 0
    assert bg_line in log_image[:i_image]
    assert bg_line in log_image[i_image:]
    assert text == text_ref
    assert image_text == image_ref

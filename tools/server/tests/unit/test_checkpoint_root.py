import os
import shutil
import tempfile
import pytest
from utils import *

# a hybrid model: a cached prompt is restored only up to a checkpoint
pytestmark = pytest.mark.skipif(not os.environ.get("LLAMA_TEST_HYBRID_MODEL"), reason="needs LLAMA_TEST_HYBRID_MODEL, a small hybrid/recurrent GGUF")

server: ServerProcess

SYSTEM = "You are an assistant. Rules:\n" + "\n".join(f"{k}. When module m{k} changes, run test t{k % 13} and update doc d{k % 7}." for k in range(60))


def make_server(**kw):
    srv = ServerProcess()
    srv.model_hf_repo = None
    srv.model_hf_file = None
    srv.model_file = os.environ["LLAMA_TEST_HYBRID_MODEL"]
    srv.n_ctx = 32768
    srv.n_slots = 1
    srv.n_predict = 8
    srv.temperature = 0.0
    srv.jinja = True
    srv.cache_ram = 512
    for k, v in kw.items():
        setattr(srv, k, v)
    return srv


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = make_server()
    yield
    server.stop()


def chat(messages, id_slot=-1):
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": messages, "max_tokens": 8, "temperature": 0.0, "id_slot": id_slot,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert res.status_code == 200
    return res.body["timings"]["cache_n"], {"role": "assistant", "content": res.body["choices"][0]["message"]["content"] or ""}


def n_system_tokens():
    # the tokens of the prompt up to the first user message
    res = server.make_request("POST", "/apply-template", data={"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x"}]})
    prompt = res.body["prompt"]
    head = prompt[:prompt.rindex("x")]
    head = head[:head.rindex("<|im_start|>")] if "<|im_start|>" in head else head
    res = server.make_request("POST", "/tokenize", data={"content": head, "add_special": False, "parse_special": True})
    return len(res.body["tokens"])


def long_chat(n_turns):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"Turn {t}: " + " ".join(f"what does m{(t * 7 + k) % 60} need?" for k in range(30))})
        _, reply = chat(msgs)
        msgs.append(reply)
    return msgs


@pytest.mark.parametrize("root", [False, True])
def test_new_chat_in_the_other_slot(root):
    # a new chat with the same system prompt in a slot that never held it: the root prompt comes from the cache
    global server
    server = make_server(n_slots=2, checkpoint_root=256 if root else None)
    server.start()
    n_sys = n_system_tokens()
    chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Which tests does m5 need?"}], id_slot=0)
    cache_n, _ = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Which docs does m9 need?"}], id_slot=1)
    if root:
        assert cache_n >= n_sys - 4
    else:
        assert cache_n == 0


@pytest.mark.parametrize("root", [False, True])
def test_new_chat_after_the_checkpoints_rotated(root):
    # a long chat fills the checkpoints of its slot; the root survives the rotation and a new chat starts after it
    global server
    server = make_server(ctx_checkpoints=3, checkpoint_min_step=0, checkpoint_root=256 if root else None)
    server.start()
    n_sys = n_system_tokens()
    long_chat(6)
    cache_n, _ = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Which docs does m9 need?"}])
    if root:
        assert cache_n >= n_sys - 4
    else:
        assert cache_n < n_sys - 4


def test_root_after_restart():
    # the root prompt is stored on the disk level as a file of its own and serves a new chat after a restart
    global server
    cache_dir = tempfile.mkdtemp(prefix='checkpoint-root-')
    try:
        server = make_server(checkpoint_root=256, cache_disk=cache_dir)
        server.start()
        n_sys = n_system_tokens()
        long_chat(2)
        server.stop()
        assert any(os.path.basename(f).split("-")[2].startswith("r") for f in os.listdir(cache_dir) if f.endswith(".bin")) or \
               any(f.split("-")[2].startswith("r") for f in os.listdir(cache_dir) if f.startswith("pc-") and f.endswith(".bin"))
        server = make_server(checkpoint_root=256, cache_disk=cache_dir)
        server.start()
        cache_n, _ = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Which docs does m9 need?"}])
        assert cache_n >= n_sys - 4
    finally:
        server.stop()
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.parametrize("save_min", [None, 64])
def test_conversation_back_after_another_chat(save_min):
    # chat B takes the slot of chat A and keeps most of it (the system prompt): with --cache-save-min chat A is saved
    # first and comes back whole
    global server
    server = make_server(checkpoint_root=256, cache_save_min=save_min)
    server.start()
    a = long_chat(3)
    a_len = server.make_request("POST", "/apply-template", data={"messages": a}).body["prompt"]
    n_a = len(server.make_request("POST", "/tokenize", data={"content": a_len, "add_special": False, "parse_special": True}).body["tokens"])
    chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Which docs does m9 need?"}])
    a.append({"role": "user", "content": "And m11?"})
    cache_n, _ = chat(a)
    if save_min:
        # restored up to the checkpoint at the end of the last prompt of chat A
        assert cache_n >= n_a - 32
    else:
        assert cache_n < n_a - 64


@pytest.mark.parametrize("spread", [False, True])
def test_edit_far_back(spread):
    # an agent rewrites a turn far back in a long chat: with --checkpoint-spread a checkpoint before it is still there
    global server
    # 6 checkpoints: the root and 5 for the turns; first-in-first-out keeps only the last 5 turns
    server = make_server(ctx_checkpoints=6, checkpoint_min_step=0, checkpoint_root=256, checkpoint_spread=spread)
    server.start()
    n_sys = n_system_tokens()
    msgs = long_chat(12)
    # rewrite the user message of turn 5 (index 1 + 2*5) and resend everything up to the end
    msgs[11] = {"role": "user", "content": "Turn 5, rewritten: what does m1 need?"}
    cache_n, _ = chat(msgs)
    if spread:
        assert cache_n > n_sys + 64
    else:
        assert cache_n <= n_sys + 8

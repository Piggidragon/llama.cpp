// llama-draft-vocab: the token ids for --spec-draft-vocab, from the token frequencies of a text corpus
//
// The most frequent tokens of the files that together cover --coverage of the token occurrences, plus every control
// and end-of-generation token, written as ids one per line. Use text like the one the model will write (code, the
// languages of the answers, the format of tool calls): a token outside the list is never drafted.

#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL -f FILE [-f FILE ...] [--coverage P] [--max-tokens N] [-o OUT]\n"
        "\n"
        "  -m MODEL          the model (only its vocabulary is loaded)\n"
        "  -f FILE           a text file of the corpus (several allowed)\n"
        "  --coverage P      share of the token occurrences the list covers (default: 0.995)\n"
        "  --max-tokens N    at most N tokens besides the control tokens (default: no limit)\n"
        "  -o OUT            output file (default: stdout)\n", argv0);
}

int main(int argc, char ** argv) {
    std::string path_model;
    std::string path_out;
    std::vector<std::string> files;
    double  coverage   = 0.995;
    int64_t max_tokens = -1;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") {
            path_model = next();
        } else if (arg == "-f" || arg == "--file") {
            files.push_back(next());
        } else if (arg == "--coverage") {
            coverage = atof(next());
        } else if (arg == "--max-tokens") {
            max_tokens = atoll(next());
        } else if (arg == "-o" || arg == "--output") {
            path_out = next();
        } else {
            usage(argv[0]);
            return arg == "-h" || arg == "--help" ? 0 : 1;
        }
    }
    if (path_model.empty() || files.empty() || !(coverage > 0.0 && coverage <= 1.0)) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path_model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load the vocabulary of '%s'\n", path_model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<uint64_t> counts(n_vocab, 0);
    uint64_t n_total = 0;

    // in pieces of whole lines, so that a large file does not take its whole token list at once
    std::vector<llama_token> tokens;
    auto count = [&](const std::string & text) {
        if (text.empty()) {
            return;
        }
        int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, true);
        tokens.resize(n < 0 ? -n : n);
        n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, true);
        for (int32_t i = 0; i < n; i++) {
            counts[tokens[i]]++;
        }
        n_total += n > 0 ? n : 0;
    };
    for (const auto & path : files) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            fprintf(stderr, "cannot open '%s'\n", path.c_str());
            return 1;
        }
        std::string piece;
        std::string line;
        while (std::getline(in, line)) {
            piece += line;
            piece += '\n';
            if (piece.size() >= (1u << 20)) {
                count(piece);
                piece.clear();
            }
        }
        count(piece);
    }
    if (n_total == 0) {
        fprintf(stderr, "the corpus has no tokens\n");
        return 1;
    }

    std::vector<llama_token> order(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](llama_token a, llama_token b) { return counts[a] > counts[b]; });

    std::vector<bool> keep(n_vocab, false);
    uint64_t n_covered = 0;
    int64_t  n_kept    = 0;
    for (llama_token id : order) {
        if (counts[id] == 0 || (double) n_covered >= coverage*n_total || (max_tokens >= 0 && n_kept >= max_tokens)) {
            break;
        }
        keep[id] = true;
        n_covered += counts[id];
        n_kept++;
    }
    // a draft can end a turn or emit the markup tokens of the template
    int64_t n_control = 0;
    for (int32_t id = 0; id < n_vocab; id++) {
        if (!keep[id] && (llama_vocab_is_control(vocab, id) || llama_vocab_is_eog(vocab, id))) {
            keep[id] = true;
            n_control++;
        }
    }

    std::ostringstream out;
    out << "# llama-draft-vocab: " << (n_kept + n_control) << " of " << n_vocab << " tokens of " << path_model << "\n";
    out << "# " << n_kept << " cover " << 100.0*n_covered/n_total << "% of the " << n_total << " tokens of the corpus, "
        << n_control << " control and end-of-generation tokens\n";
    for (int32_t id = 0; id < n_vocab; id++) {
        if (keep[id]) {
            out << id << "\n";
        }
    }
    if (path_out.empty()) {
        fputs(out.str().c_str(), stdout);
    } else {
        std::ofstream f(path_out);
        f << out.str();
        if (!f) {
            fprintf(stderr, "cannot write '%s'\n", path_out.c_str());
            return 1;
        }
    }
    fprintf(stderr, "%" PRId64 " tokens (%.1f%% of the vocabulary) cover %.2f%% of %" PRIu64 " corpus tokens, plus %" PRId64 " control tokens\n",
            n_kept, 100.0*n_kept/n_vocab, 100.0*n_covered/n_total, n_total, n_control);

    llama_model_free(model);
    return 0;
}

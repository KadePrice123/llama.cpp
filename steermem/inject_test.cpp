// steermem: does llama_inject_set change the model's output, and only where it is told to?
//
// Decodes a prompt three times: plain; with a random unit vector injected at
// --layer on ONE prompt position; and cleared again. The last-token logits must
// differ in the middle run and match in the first and third. Prints the max
// absolute logit difference for each pair and INJECT-TEST PASS/FAIL.
//
//   g++ -O2 -std=c++17 -I include -I ggml/include steermem/inject_test.cpp \
//       -L build-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,build-cpu/bin \
//       -o build-cpu/bin/steermem-inject-test
//   build-cpu/bin/steermem-inject-test MODEL.gguf [layer=8] [pos=3] [scale=0.5] [threads=4]
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static std::vector<float> last_logits(llama_context * ctx, llama_model * model, const std::vector<llama_token> & toks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode rc %d\n", rc);
        exit(2);
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * p = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(p, p + n_vocab);
}

static float maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static int argmax(const std::vector<float> & a) {
    int k = 0;
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] > a[k]) {
            k = (int) i;
        }
    }
    return k;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [layer] [pos] [scale] [threads]\n", argv[0]);
        return 1;
    }
    const int   layer   = argc > 2 ? atoi(argv[2]) : 8;
    const int   pos     = argc > 3 ? atoi(argv[3]) : 3;
    const float scale   = argc > 4 ? (float) atof(argv[4]) : 0.5f;
    const int   threads = argc > 5 ? atoi(argv[5]) : 4;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "model failed to load\n");
        return 2;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context failed\n");
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = "Recall each of these from memory, in order:\n1. James 3:14\nAnswer:\n";
    std::vector<llama_token> toks(128);
    const int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, true);
    if (n <= pos) {
        fprintf(stderr, "prompt too short (%d tokens) for pos %d\n", n, pos);
        return 2;
    }
    toks.resize(n);
    const int n_embd = llama_model_n_embd(model);
    printf("model n_embd %d, prompt %d tokens, inject layer %d pos %d scale %.2f\n", n_embd, n, layer, pos, scale);

    const std::vector<float> plain = last_logits(ctx, model, toks);

    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n_embd);
    float norm = 0.0f;
    for (auto & x : v) { x = nd(rng); norm += x * x; }
    norm = std::sqrt(norm);
    for (auto & x : v) { x /= norm; }
    const llama_pos p = pos;
    if (llama_inject_set(ctx, layer, 1, &p, v.data(), scale) != 0) {
        fprintf(stderr, "llama_inject_set failed\n");
        return 2;
    }
    const std::vector<float> injected = last_logits(ctx, model, toks);
    llama_inject_clear(ctx);
    const std::vector<float> cleared = last_logits(ctx, model, toks);

    const float d1 = maxdiff(plain, injected);
    const float d2 = maxdiff(plain, cleared);
    printf("max |logit diff|  plain vs injected: %.4f   plain vs cleared: %.4f\n", d1, d2);
    printf("argmax plain %d  injected %d  cleared %d\n", argmax(plain), argmax(injected), argmax(cleared));
    const bool ok = d1 > 1e-3f && d2 < 1e-3f;
    printf("INJECT-TEST %s\n", ok ? "PASS" : "FAIL");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}

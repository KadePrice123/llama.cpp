#include "ngram-bias.h"

#include "ngram-cache.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct common_ngram_bias_ctx {
    common_ngram_cache nc_static;   // from llama-lookup-create, the swappable table
    common_ngram_cache nc_dynamic;  // optional, previous generations
    common_ngram_cache nc_context;  // built from this generation as it proceeds

    std::vector<llama_token> inp;     // rolling token history
    std::vector<llama_token> target;  // per-request target, if any

    float bias        = 0.0f;
    float init_bias   = 0.0f;       // strength for STARTING a target (see apply_target)
    int   after       = 0;          // stay inert for this many generated tokens
    bool  fired       = false;      // target fully walked once; latch off
    float gap         = 0.0f;       // informativeness gate; 0 = always bias
    bool  use_context = false;      // feed generation back in? see accept()
    int   ngram_min   = LLAMA_NGRAM_MIN;
    int   ngram_max   = LLAMA_NGRAM_MAX;

    size_t  n_ctx_synced = 0;       // how much of inp is already folded into nc_context
    int64_t n_hits       = 0;
    int64_t n_steps      = 0;
    int64_t n_skipped    = 0;       // matched, but the model was already going there
};

static const char * common_ngram_bias_name(const struct llama_sampler * /*smpl*/) {
    return "ngram-bias";
}

static void common_ngram_bias_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_ngram_bias_ctx *) smpl->ctx;

    ctx->inp.push_back(token);

    // The context cache is OFF by default and that default is load-bearing.
    //
    // common_ngram_cache_draft tries nc_context FIRST and with lax thresholds,
    // so anything the model has just repeated outranks the static table. For
    // speculative decoding that is correct -- predicting repetition is free
    // speed. For biasing it is a runaway feedback loop: measured, the model
    // emitted "0.000", the context cache learned 0->0, and 307 of 369 boosts
    // went to the token '0', locking it into an infinite zero loop while the
    // static table's payload was boosted exactly once.
    //
    // Biasing must steer AWAY from what the model is already doing. Feeding its
    // own output back in does the opposite.
    const int nnew = (int) (ctx->inp.size() - ctx->n_ctx_synced);
    if (ctx->use_context && nnew > 0 && (int) ctx->inp.size() > ctx->ngram_max) {
        common_ngram_cache_update(ctx->nc_context, ctx->ngram_min, ctx->ngram_max,
                                  ctx->inp, nnew, /*print_progress =*/ false);
        ctx->n_ctx_synced = ctx->inp.size();
    }
}

// Add the bias to one token, subject to the informativeness gate.
//
// The gate exists because biasing on MATCH is nearly useless. Measured without
// it: over one reasoning trace a table fired 117 times and every boost landed on
// filler ('  ', 'ler', ' manifold', ' **', ' calibration') that the model was
// emitting anyway, leaving the output byte-identical. What matters is
// DISAGREEMENT -- bias only where the table's token is meaningfully below the
// model's own top choice. gap == 0 keeps the old always-bias behaviour.
static void apply_bias(common_ngram_bias_ctx * ctx, llama_token_data_array * cur_p,
                       llama_token want) {
    if (ctx->gap > 0.0f) {
        float max_logit  = -INFINITY;
        float want_logit = -INFINITY;
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit > max_logit) {
                max_logit = cur_p->data[i].logit;
            }
            if (cur_p->data[i].id == want) {
                want_logit = cur_p->data[i].logit;
            }
        }
        if (want_logit == -INFINITY) {
            return;                   // not a candidate, or grammar-masked
        }
        if (max_logit - want_logit < ctx->gap) {
            ctx->n_skipped++;         // model already going there; leave it alone
            return;
        }
    }

    // cur_p is indexed by position, not token id, and is only id-indexable while
    // untruncated and unsorted. This sampler must run FIRST in the chain -- see
    // sampling.cpp. The fast path covers the untouched full-vocab case.
    if (!cur_p->sorted && (size_t) want < cur_p->size && cur_p->data[want].id == want) {
        LOG_INF("ngram-bias: step %lld boosting token %d by %+.2f (logit %.3f -> %.3f)\n",
                (long long) ctx->n_steps, want, ctx->bias,
                cur_p->data[want].logit, cur_p->data[want].logit + ctx->bias);
        cur_p->data[want].logit += ctx->bias;
        ctx->n_hits++;
        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].id == want) {
            cur_p->data[i].logit += ctx->bias;
            cur_p->sorted = false;    // logits changed, any prior ordering is stale
            ctx->n_hits++;
            return;
        }
    }

    // Token already pruned by an earlier sampler. Nothing to boost -- this is the
    // silent-failure case the "run first" rule exists to prevent.
}

// Table mode: ask the n-gram caches what comes next after the current history.
static void apply_table(common_ngram_bias_ctx * ctx, llama_token_data_array * cur_p) {
    // common_ngram_cache_draft asserts draft.size() == 1 on entry and expects
    // that element to be the previously sampled token. We ask for exactly one
    // continuation: the single token the table says comes next.
    std::vector<llama_token> draft = { ctx->inp.back() };
    common_ngram_cache_draft(ctx->inp, draft, /*n_draft =*/ 1,
                             ctx->ngram_min, ctx->ngram_max,
                             ctx->nc_context, ctx->nc_dynamic, ctx->nc_static);
    if (draft.size() < 2) {
        return;                       // no table match for this history
    }
    apply_bias(ctx, cur_p, draft[1]);
}

// Target mode: walk an explicit per-request string instead of a table.
//
// This is the mode that pairs with a lazy grammar. The grammar is compiled from
// the memory table, so every branch it allows is an entry that genuinely exists;
// the target is whichever branch a retriever ranked highest. The grammar says
// what is LEGAL, this says what is LIKELY, and the model still chooses between
// them -- including the abstain branch.
//
// The pairing is safe by construction rather than by ordering: llama.cpp masks
// grammar-disallowed tokens to -INFINITY, and -inf plus any finite bias is still
// -inf. Boosting a forbidden branch is a mathematical no-op, so NO bias strength
// can make the model emit something that is not in the table. That is the whole
// reason this mode is worth having: every earlier failure in this project came
// from biasing toward something the model could not verify, and here it can only
// ever be pointed at an entry that exists.
//
// Matching is the longest suffix of history against the longest prefix of the
// target, so the bias tracks the model along the target and points at whatever
// comes next. k == 0 is deliberately NOT biased: before the model has emitted
// any of the target, boosting its first token would push it to start reciting
// at every position in free text. Once a lazy grammar fires it forces the
// target's opening tokens anyway, which carries k to 1 on its own.
static void apply_target(common_ngram_bias_ctx * ctx, llama_token_data_array * cur_p) {
    // FIRE ONCE. After the target has been walked to the end, the generated suffix
    // stops matching it, `best` falls back to 0, and init_bias happily starts the
    // whole retrieval again -- measured: the same lookup emitted 3-4 times in one
    // answer, each introduced by "Wait,". One retrieval is the whole intent, so
    // latch off once it has happened.
    if (ctx->fired) {
        return;
    }

    const size_t tn = ctx->target.size();
    const size_t in = ctx->inp.size();
    if (tn == 0) {
        return;
    }
    if (in == 0) {
        // Nothing generated yet. Still a valid place to initiate.
        if (ctx->init_bias > 0.0f) {
            const float saved = ctx->bias;
            ctx->bias = ctx->init_bias;
            apply_bias(ctx, cur_p, ctx->target[0]);
            ctx->bias = saved;
        }
        return;
    }

    size_t best = 0;
    for (size_t k = std::min(tn, in); k >= 1; --k) {
        bool ok = true;
        for (size_t j = 0; j < k; ++j) {
            if (ctx->inp[in - k + j] != ctx->target[j]) { ok = false; break; }
        }
        if (ok) { best = k; break; }
    }

    if (best >= tn) {
        ctx->fired = true;            // walked it end to end; do not start another
        return;
    }

    if (best == 0) {
        // NOT on the target yet. Biasing the target's FIRST token here is what
        // lets the bias *initiate* a retrieval instead of only following one the
        // model already began. Without it the sampler is inert whenever the model
        // does not spontaneously start the phrase -- measured: identical output at
        // bias 0, 6 and 12, because the walk could never get off the ground.
        //
        // It is a separate, normally much smaller strength because it applies at
        // EVERY position in free text, where the grammar is not yet active and so
        // nothing bounds it. Too high and the model opens every sentence with the
        // trigger word. 0 disables initiation and restores follow-only behaviour.
        if (ctx->init_bias > 0.0f) {
            const float saved = ctx->bias;
            ctx->bias = ctx->init_bias;
            apply_bias(ctx, cur_p, ctx->target[0]);
            ctx->bias = saved;
        }
        return;
    }
    apply_bias(ctx, cur_p, ctx->target[best]);
}

static void common_ngram_bias_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_ngram_bias_ctx *) smpl->ctx;

    ctx->n_steps++;

    // POSITIONAL GATE. Reasoning models open with a fixed structural habit --
    // Qwen3.5 writes "1.  **Analyze the Request:**" in 6/6 samples -- and fighting
    // it costs ~20 logits, which is enough force to damage the surrounding text.
    // Later steps are far less determined (step 2 came back as three different
    // headings across the same samples, one of them already a recall), so that is
    // where steering is cheap.
    //
    // A short structural anchor cannot express "later": a target beginning "\n\n2."
    // also matches the blank line before step 1, so the bias fires early and pushes
    // a 2. where the 1. belongs -- measured, step 1 corrupted at bias 14+. Position
    // is the thing that actually distinguishes them, so gate on it directly.
    if (ctx->after > 0 && (int) ctx->inp.size() < ctx->after) {
        return;
    }

    // inp may legitimately be empty: with a target and init_bias set we want to
    // be able to initiate a retrieval on the very first generated token.
    if (cur_p->size == 0) {
        return;
    }
    if (ctx->bias == 0.0f && ctx->init_bias == 0.0f) {
        return;
    }
    if (ctx->inp.empty() && ctx->target.empty()) {
        return;                       // table lookups need history; targets do not
    }

    // An explicit per-request target wins over the table: the caller has already
    // decided what it wants promoted, so a table lookup would only add noise.
    if (!ctx->target.empty()) {
        apply_target(ctx, cur_p);
        return;
    }
    apply_table(ctx, cur_p);
}

static void common_ngram_bias_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_ngram_bias_ctx *) smpl->ctx;

    ctx->inp.clear();
    ctx->nc_context.clear();
    ctx->n_ctx_synced = 0;
}

static void common_ngram_bias_free(struct llama_sampler * smpl) {
    delete (common_ngram_bias_ctx *) smpl->ctx;
}

// Backend-sampling hooks are intentionally left null: this sampler runs on the
// CPU side of the chain. llama.cpp treats the null backend_init as "unsupported"
// and keeps CPU sampling, which is what we want.
static struct llama_sampler_i common_ngram_bias_i = {
    /* .name              = */ common_ngram_bias_name,
    /* .accept            = */ common_ngram_bias_accept,
    /* .apply             = */ common_ngram_bias_apply,
    /* .reset             = */ common_ngram_bias_reset,
    /* .clone             = */ nullptr,
    /* .free              = */ common_ngram_bias_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct llama_sampler * common_ngram_bias_init(
        const std::string & path_static,
        const std::string & path_dynamic,
        const std::vector<llama_token> & target,
        float bias,
        float init_bias,
        int   after,
        float gap,
        bool  use_context,
        int   ngram_min,
        int   ngram_max) {
    if (path_static.empty() && path_dynamic.empty() && target.empty()) {
        LOG_WRN("%s: no lookup table and no target given, ngram-bias disabled\n", __func__);
        return nullptr;
    }

    auto * ctx = new common_ngram_bias_ctx();
    ctx->bias        = bias;
    ctx->init_bias   = init_bias;
    ctx->after       = after;
    ctx->gap         = gap;
    ctx->use_context = use_context;
    ctx->target      = target;
    ctx->ngram_min   = ngram_min < LLAMA_NGRAM_MIN ? LLAMA_NGRAM_MIN : ngram_min;
    ctx->ngram_max   = ngram_max > LLAMA_NGRAM_MAX ? LLAMA_NGRAM_MAX : ngram_max;

    if (!path_static.empty()) {
        try {
            ctx->nc_static = common_ngram_cache_load(path_static);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to load static lookup table '%s': %s\n",
                    __func__, path_static.c_str(), e.what());
            delete ctx;
            return nullptr;
        }
    }

    if (!path_dynamic.empty()) {
        try {
            ctx->nc_dynamic = common_ngram_cache_load(path_dynamic);
        } catch (const std::exception & e) {
            LOG_WRN("%s: failed to load dynamic lookup table '%s': %s (continuing)\n",
                    __func__, path_dynamic.c_str(), e.what());
        }
    }

    // Logged loudly on purpose. This sampler changes what the model says, and a
    // silently-active steering layer is exactly the thing you do not want to
    // discover later while debugging a wrong answer.
    LOG_INF("%s: ngram-bias ACTIVE: bias=%.2f gap=%.2f context=%s mode=%s "
            "target=%zu tok ngram=%d..%d static=%zu dynamic=%zu\n",
            __func__, bias, gap, use_context ? "on" : "off",
            target.empty() ? "table" : "target", target.size(),
            ctx->ngram_min, ctx->ngram_max,
            ctx->nc_static.size(), ctx->nc_dynamic.size());

    return llama_sampler_init(&common_ngram_bias_i, ctx);
}

void common_ngram_bias_stats(const struct llama_sampler * smpl, int64_t & n_hits, int64_t & n_steps) {
    n_hits  = 0;
    n_steps = 0;
    if (!smpl || smpl->iface != &common_ngram_bias_i) {
        return;
    }
    const auto * ctx = (const common_ngram_bias_ctx *) smpl->ctx;
    n_hits  = ctx->n_hits;
    n_steps = ctx->n_steps;
}

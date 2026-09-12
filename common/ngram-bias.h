#pragma once

// N-gram lookup BIAS sampler.
//
// This is deliberately NOT speculative decoding. `--spec-type ngram-cache` uses
// the same lookup tables to DRAFT tokens which the model then verifies, and that
// verification step is provably distribution-preserving: a table entry the model
// would not have produced on its own is rejected, every time. Measured on this
// box: an invented fact loaded into a static cache and drafted at 62-80%
// acceptance was still never emitted, because the model's own distribution did
// not support it.
//
// This sampler instead ADDS a bias to the logit of the token the table expects
// next. There is no verification, so the table can genuinely steer output toward
// facts the model does not have. That is the point, and it is also the cost:
// if the table is wrong the model states the wrong fact with full confidence.
//
// The bias must be SEQUENCE-CONDITIONAL. A static per-token bias on the tokens
// of "74813" (five separate digit tokens) does not produce "74813" -- it makes
// every one of those digits more likely at every position and collapses to
// "1111111111". Only biasing the single token the table expects NEXT, given what
// has already been emitted, produces the intended string. Hence accept() keeping
// a rolling history and apply() looking up exactly one continuation token.
//
// Cost is one hash lookup and one array write per generated token, against a
// forward pass of ~25 ms. The equivalent done over HTTP (one request per token,
// re-prefilling because this model has no prompt caching) measured ~30x slower;
// that is the whole reason this lives in the sampler chain.

#include "llama.h"

#include <string>
#include <vector>

// Create the sampler. Returns nullptr if no table could be loaded.
//
//   path_static  : cache built by llama-lookup-create (required)
//   path_dynamic : optional cache of previous generations
//   bias         : added to the expected token's logit. Measured on Qwen3.5-4B,
//                  a 5-digit fact the model could not know needed ~5.0 to be
//                  emitted intact; 2-3 produced partial strings.
//   ngram_min/max: history lengths to match on (see LLAMA_NGRAM_MIN/MAX)
//   gap          : informativeness gate. Only bias when the table's expected
//                  token is at least `gap` logits BELOW the model's own top
//                  choice -- i.e. only where the table disagrees. 0 disables the
//                  gate. This matters more than it sounds: measured without it,
//                  a table fired 117 times in one reasoning trace and every
//                  boost landed on filler ('  ', ' manifold', ' **') that the
//                  model was emitting anyway, leaving output byte-identical.
//   use_context  : feed the model's own generation back into the lookup. OFF by
//                  default and it should usually stay off -- the context cache
//                  is consulted FIRST and with lax thresholds, so it amplifies
//                  whatever the model just repeated. Measured with it on: the
//                  model emitted "0.000", the cache learned 0->0, and 307 of 369
//                  boosts went to '0' in a runaway loop while the real payload
//                  was boosted once. Biasing must steer AWAY from what the model
//                  is already doing.
//   target       : optional PER-REQUEST target token sequence. When set it
//                  replaces the table lookup: the sampler tracks the model along
//                  this sequence and biases whatever comes next.
//
//                  This is the mode that pairs with a lazy grammar. Compile the
//                  grammar from the memory table (so every legal branch is an
//                  entry that exists), then pass the branch a retriever ranked
//                  highest as the target. The grammar says what is LEGAL, the
//                  target says what is LIKELY, and the model still chooses --
//                  including the abstain branch.
//
//                  Safe by construction, not by ordering: llama.cpp masks
//                  grammar-disallowed tokens to -INFINITY, and -inf plus any
//                  finite bias is still -inf. No bias strength can make the
//                  model emit something outside the table.
//   init_bias    : strength used to bias the target's FIRST token when the model
//                  is not already walking it. This is what lets the bias START a
//                  retrieval rather than only follow one the model happened to
//                  begin -- without it the sampler is inert unless the model
//                  spontaneously writes the trigger, which an un-fine-tuned model
//                  will not do. Applies at every position in free text, where no
//                  grammar bounds it, so keep it well below `bias`. 0 disables.
//   after        : stay inert until this many tokens have been generated, so the
//                  model's fixed opening habit is left alone and steering happens
//                  where its prior is weak. Position is the only thing that
//                  distinguishes 'later in the reasoning' from 'in the heading':
//                  a short structural anchor cannot: a target beginning with a
//                  blank line and "2." also matches the seam before step 1, so
//                  the bias fires early and pushes a 2. where the 1. belongs --
//                  measured, step 1 corrupted at bias 14+. 0 = active at once.
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
        int   ngram_max);

// How many tokens were biased, and how many sampling steps ran. Useful for
// telling "the table never matched" apart from "the bias was too weak", which
// otherwise look identical from the outside.
void common_ngram_bias_stats(const struct llama_sampler * smpl, int64_t & n_hits, int64_t & n_steps);

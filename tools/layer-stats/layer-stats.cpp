/**
 *
 * llama-layer-stats
 *
 *
 * Measure the hidden state between layers during inference. Average over all tokens.
 *
 * Report three metrics per layer:
 *
 * - cos_sim :  cosine similarity of    l_out[N] vs. l_out[N-1 ]   (per token, averaged)
 * - l2_dist :  L2 norm distance      ||l_out[N]  -  l_out[N-1]||  (per token, averaged)
 * - l2_norm :  L2 norm               ||l_out[N]||                 (per token, averaged)
 *
**/

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


// cosine similarity between two equal-length vectors
static float cossim(const float * a, const float * b, int64_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0.0 ? (float) (dot / den) : 0.0f;
}

// L2 norm distance between two equal-length vectors
static float l2_distance(const float * a, const float * b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        s += d * d;
    }
    return (float) std::sqrt(s);
}

// L2 norm of a vector
static float l2_norm(const float * a, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += (double) a[i] * (double) a[i];
    }
    return (float) std::sqrt(s);
}

//
// backend-agnostic tensor data access helpers
//

struct tensor_data_view {
    tensor_data_view(void * p) : ptr_(p), buf_({}) {}
    tensor_data_view(std::vector<char> && buf) : ptr_(buf.data()), buf_(std::move(buf)) {}
    inline void * data() const { return ptr_; }

    private:
        void * ptr_;
        std::vector<char> buf_;
};

static tensor_data_view tensor_get_view(const ggml_tensor * t) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        return tensor_data_view(t->data);
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<char> buf(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    return tensor_data_view(std::move(buf));
}

// per-layer accumulated metrics
struct layer_metrics {
    int64_t count = 0;
    double sum_cos_sim = 0.0;
    double sum_l2_dist = 0.0;
    double sum_l2_norm = 0.0;
    bool has_prev = false;
};

// session state passed to the callback
struct cb_data {
    int n_layer = 0; // set once we know the model depth // TODO: is this really needed?
    std::vector<layer_metrics> metrics; // index by layer (length n_layer)

    // previous l_out hidden states, one flat f32 buffer per layer:
    // layout:
    //         [n_tokens][n_embd], row-major, contiguous F32
    //
    // keyed by layer so we pair l_out[N] with l_out[N-1] of the SAME decode
    // prev[layer] holds l_out[layer-1]? see TODO below
    std::vector<std::vector<float>> prev;
    std::vector<bool>               prev_valid;

    // scratch reused across observations to avoid realloc
    std::vector<float> scratch;
};

// return layer index from tensor name "l_out-{li}"
//
// TODO: support "l_last-{li}" for mHC models (and other names?)
static int extract_layer(const std::string & name) {
    static const std::string key = "l_out-";
    if (name.rfind(key, 0) != 0) {
        return -1;
    }
    try {
        return std::stoi(name.substr(key.size()));
    } catch (...) {
        return -1;
    }
}

static bool tensor_is_layer_output(const ggml_tensor * t) {
    if (t == nullptr || ggml_is_empty(t)) {
        return false;
    }
    return extract_layer(t->name) >= 0;
}

static bool cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * data = static_cast<cb_data *>(user_data);

    if (ask) {
        // scheduler is asking if we want this node; only take l_out
        return tensor_is_layer_output(t);
    }

    const int layer = extract_layer(t->name);

    if (layer < 0 || layer >= data->n_layer) { // should not happen
        return true;
    }

    // hidden state layout: [n_embd, n_tokens]  (ne[0]=n_embd fastest, ne[1]=n_tokens)
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    // TODO: guard against unexpected rank/type. Expect 2D F32 [n_embd, n_tokens].
    //       If a model emits a different type we should cast (see tensor-debug's switch).

    const tensor_data_view view = tensor_get_view(t);
    const float * cur = (const float *) view.data();

    auto & m = data->metrics[layer];

    // L2 norm of this layer's output, averaged over tokens
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        const float * c = cur + tok * n_embd;
        m.sum_l2_norm += (double) l2_norm(c, n_embd);
        m.count += 1;
    }

    // pair against the previous layer's output (same token positions, same decode)
    if (layer > 0 && data->prev_valid[layer - 1]) {
        const std::vector<float> & prev = data->prev[layer - 1];
        // shapes must match; if a decode produced a different n_tokens, skip pairing
        if ((int64_t) prev.size() == n_embd * n_tokens) {
            for (int64_t tok = 0; tok < n_tokens; ++tok) {
                const float * c = cur + tok * n_embd;
                const float * p = prev.data() + tok * n_embd;
                m.sum_cos_sim  += (double) cossim(c, p, n_embd);
                m.sum_l2_dist += (double) l2_distance(c, p, n_embd);
            }
            // count already advanced above for mag
        }
    }

    // stash current as the "previous" for the next layer to compare against.
    // layers fire 0..n_layer-1 in order within a decode, so prev[layer] = cur here
    // and layer N reads prev[layer-1]. Reset handled by ... (TODO: decode boundary)
    data->prev[layer].assign(cur, cur + n_embd * n_tokens);
    data->prev_valid[layer] = true;

    return true;
}

static void print_report(const cb_data & data) {

    struct row {
        int layer;
        double cossim;
        double l2_dist;
        double l2_mag;
    };

    std::vector<row> rows;

    for (int i = 0; i < data.n_layer; ++i) {
        const auto & m = data.metrics[i];
        if (m.count == 0) {
            continue;
        }
        rows.push_back({
            i, m.sum_cos_sim / static_cast<double>(m.count),
               m.sum_l2_dist / static_cast<double>(m.count),
               m.sum_l2_norm / static_cast<double>(m.count),
        });
    }

    // TODO: sort? decide the single "importance" ordering?

    LOG_INF("\n%6s  %12s  %14s  %12s\n", "rank", "cossim", "l2_dist", "l2_mag");
    LOG_INF("========================================================\n");
    int r = 1;
    for (const auto & x : rows) {
        LOG_INF("%6d  %12.6f  %14.6f  %12.6f\n", r++, x.cossim, x.l2_dist, x.l2_mag);
    }
    LOG_INF("\n");
}

//
// driver
//
// TODO: decide input shape: single -p prompt, or -f calibration file chunked like
//       imatrix's compute_imatrix? For a first cut, one prompt / one decode is enough.
static bool run_layer_cossim(llama_context * ctx, const std::string & prompt) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    LOG_INF("%s: tokenizing prompt ...\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos, /*parse_special=*/false);

    const int n_batch = llama_n_batch(ctx);
    if ((int) tokens.size() > n_batch) {
        tokens.resize(n_batch); // TODO: support multi-batch; for now cap at n_batch
    }

    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    for (int i = 0; i < (int) tokens.size(); ++i) {
        batch.token[i]  = tokens[i];
        batch.pos[i]    = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = (int) tokens.size();

    LOG_INF("%s: decoding %d tokens ...\n", __func__, batch.n_tokens);
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        LOG_ERR("%s: llama_decode failed (%d)\n", __func__, ret);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LAYER_STATS, print_usage)) {
        return EXIT_FAILURE;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: no prompt provided; please specify `-p / --prompt` or `-f / --file`\n", __func__);
        return EXIT_FAILURE;
    }

    // no warmup: it would fire the callback on an empty run
    params.warmup = false;

    // session state lives on main's stack; must outlive llama_decode
    cb_data state;

    params.cb_eval            = cb;
    params.cb_eval_user_data  = &state;

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return EXIT_FAILURE;
    }

    // size the per-layer containers now that we know the depth
    // TODO: get n_layer robustly (llama_model_n_layers / hparams); confirm API name
    state.n_layer  = 32; // FIXME: replace with real n_layer
    state.metrics.assign(state.n_layer, {});
    state.prev.assign(state.n_layer, {});
    state.prev_valid.assign(state.n_layer, false);

    const bool ok = run_layer_cossim(ctx, params.prompt);

    if (ok) {
        print_report(state);
    }

    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

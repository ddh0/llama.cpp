/**
 * llama-layer-stats
 *
 * TODO: support `l_last` for mHC models
**/

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <map>
#include <string>
#include <vector>

//
// per-layer accumulated statistics
//

// running sums for a single layer; every field is summed over observed tokens
struct layer_data_t {
    // pairwise (vs previous layer / input embedding)
    double sum_cos_sim = 0.0; // sum of cos(h_N, h_{N-1}) per token
    double sum_l2_dist = 0.0; // sum of ||h_N - h_{N-1}||_2 per token

    // self (vs zero)
    double sum_l2_norm = 0.0; // sum of ||h_N||_2 per token
    double sum_l1_norm = 0.0; // sum of ||h_N||_1 per token
    double sum_inf_norm = 0.0; // sum of ||h_N||_inf per token

    int64_t n_tokens = 0; // number of tokens these sums are accumulated over
};

struct cb_data_t {
    int n_layer = 0; // number of layers in the model (for bounds / sanity checks)

    // per-layer running stats, indexed by layer number
    std::map<int, layer_data_t> stats;

    // previous layer's hidden state, per token: prev[layer] holds h_{layer-1}
    // laid out as [n_embd] contiguous floats per token
    std::map<int, std::vector<float>> prev;

    // TODO: measure against model input?
    bool have_input = false;
};

static float cos_sim(const float * a, const float * b, int64_t n) {
    double dot = 0.0;
    double na  = 0.0;
    double nb  = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0f; // handle this in the caller!
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

static float l2_norm(const float * x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    return static_cast<float>(std::sqrt(s));
}

static float l2_dist(const float * a, const float * b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}

static float l1_norm(const float * x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += std::fabs(static_cast<double>(x[i]));
    }
    return static_cast<float>(s);
}

static float inf_norm(const float * x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s = std::max(s, std::fabs(static_cast<double>(x[i])));
    }
    return static_cast<float>(s);
}

// tensor data helper
static const float * tensor_get_f32(const ggml_tensor * t, std::vector<float> & scratch) {
    const size_t nbytes = ggml_nbytes(t);

    if (ggml_backend_buffer_is_host(t->buffer)) {
        return static_cast<const float *>(t->data);
    }

    scratch.resize(nbytes / sizeof(float));
    ggml_backend_tensor_get(t, scratch.data(), 0, nbytes);
    return scratch.data();
}

// filter: only ask for the layer-output node.
// TODO: extend match set to "l_last" to support mHC models
static bool is_layer_output(const ggml_tensor * t) {
    if (t == nullptr || t->name[0] == '\0') {
        return false;
    }

    // node names are formatted `%s-%d` (name, layer index), e.g. "l_out-19"
    // for layer 20 (0-indexed). Match the "l_out-" prefix and require at least
    // one trailing digit.
    const char * prefix = "l_out-";
    const size_t plen   = strlen(prefix);
    if (strncmp(t->name, prefix, plen) != 0) {
        return false;
    }
    return t->name[plen] != '\0' && isdigit(static_cast<unsigned char>(t->name[plen]));
}

// parse the layer index out of a `l_out-<il>` node name.
// returns -1 if the name does not match the expected format.
static int parse_layer_index(const ggml_tensor * t) {
    const char * dash = strrchr(t->name, '-');
    if (dash == nullptr || dash == t->name) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(dash + 1, &end, 10);
    if (end == dash + 1 || *end != '\0') {
        return -1;
    }
    return static_cast<int>(il);
}

// the scheduler calls with ask=true to ask if we want this tensor, then with
// ask=false to let us read it.
static bool cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<cb_data_t *>(user_data);

    if (ask) {
        return is_layer_output(t);
    }

    const int layer = parse_layer_index(t);
    if (layer < 0) {
        LOG_ERR("%s: could not parse layer index from tensor name '%s'\n", __func__, t->name);
        return true;
    }
    if (state->n_layer > 0 && layer >= state->n_layer) {
        LOG_ERR("%s: layer index %d out of range (n_layer=%d)\n", __func__, layer, state->n_layer);
        return true;
    }

    // `l_out` shape is [n_embd, n_tokens, ...] (ne[0] = n_embd, ne[1] = n_tokens)
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    std::vector<float> scratch;
    const float * data = tensor_get_f32(t, scratch);

    auto & st = state->stats[layer];

    // per-token loop
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        const float * h = data + tok * n_embd; // current layer's token vector

        // self metrics (always computable)
        st.sum_l2_norm   += l2_norm(h, n_embd);
        st.sum_l1_norm   += l1_norm(h, n_embd);
        st.sum_inf_norm += inf_norm(h, n_embd);

        // pairwise metrics (need a previous vector)
        // layer 0 uses the input embedding as "previous" if we captured it
        const float * prev = nullptr;
        if (layer > 0) {
            auto it = state->prev.find(layer - 1);
            if (it != state->prev.end() && static_cast<int64_t>(it->second.size()) == n_embd * n_tokens) {
                prev = it->second.data() + tok * n_embd;
            }
        }

        if (prev) {
            st.sum_cos_sim += cos_sim(h, prev, n_embd);
            st.sum_l2_dist += l2_dist(h, prev, n_embd);
        }

        st.n_tokens++;
    }

    state->prev[layer].assign(data, data + n_embd * n_tokens);

    return true;
}

//
// reporting
//

struct layer_row {
    int layer;
    double cos_sim;
    double l2_dist;
    double l2;
    double l1;
    double linf;
};

static void print_report(const cb_data_t & state) {
    GGML_UNUSED(state); // TODO!
    std::vector<layer_row> rows;

    for (const auto & kv : state.stats) {
        const int layer = kv.first;
        const auto & s   = kv.second;
        if (s.n_tokens <= 0) {
            continue;
        }
        const double n = static_cast<double>(s.n_tokens);
        rows.push_back(layer_row{
            layer,
            s.sum_cos_sim / n,
            s.sum_l2_dist / n,
            s.sum_l2_norm / n,
            s.sum_l1_norm / n,
            s.sum_inf_norm / n,
        });
    }

    // sort by L2 distance to previous layer, descending (most change first)
    // TODO: make the ranking metric configurable via a CLI flag?
    std::sort(rows.begin(), rows.end(), [](const layer_row & a, const layer_row & b) {
        return a.l2_dist > b.l2_dist;
    });

    printf("\n");
    printf("%6s  %6s  %14s  %14s  %12s  %12s  %12s\n",
           "rank", "layer", "cos_sim->prev", "L2dist->prev", "L2", "L1", "Linf");
    printf("--------------------------------------------------------------------------------\n");

    int rank = 1;
    for (const auto & r : rows) {
        printf("%6d  %6d  %14.6f  %14.6f  %12.4f  %12.4f  %12.4f\n",
               rank++, r.layer, r.cos_sim, r.l2_dist, r.l2, r.l1, r.linf);
    }
    printf("\n");
}

//
// driver
//

static void run_layer_stats(llama_context * ctx, const common_params & params) {
    LOG_INF("%s: tokenizing the input ..\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    if (tokens.empty()) {
        LOG_ERR("%s: no tokens produced from prompt\n", __func__);
        return;
    }

    const int n_ctx = static_cast<int>(tokens.size());

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    for (int i = 0; i < n_ctx; ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, i == n_ctx - 1);
    }

    LOG_INF("%s: decoding %d tokens ...\n", __func__, batch.n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode failed\n", __func__);
    }

    llama_batch_free(batch);
}

static void print_usage(int argc, char ** argv) {
    GGML_UNUSED(argc); GGML_UNUSED(argv); // TODO
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();

    common_params params;
    params.warmup = false; // avoid capturing an empty warmup pass

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LAYER_STATS, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("no prompt provided: please specify -p / --prompt OR -f / --file\n");
        return 1;
    }

    cb_data_t state;

    params.cb_eval = cb;
    params.cb_eval_user_data = &state;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    // now that the model is loaded, fill in the layer count
    state.n_layer = llama_model_n_layer(model);

    run_layer_stats(ctx, params);

    print_report(state);

    llama_backend_free();
    return 0;
}

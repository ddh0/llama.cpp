/**
 *
 * llama-layer-stats
 *
 * TODO: support `l_last` for mHC models
**/

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <cctype>
#include <map>
#include <string>
#include <vector>

// global CLI option
//   0 = rank by L2 distance to previous layer, descending (most change first)
//   1 = rank by cosine similarity to previous layer, ascending (most change first)
static int g_rank_metric = 0;

// Welford's online algorithm for running mean / variance.
// Tracks per-token observations of a scalar metric; variance is over all
// observed tokens.
struct running_stats_t {
    int64_t n    = 0;     // number of observations
    double  mean = 0.0;   // running mean
    double  m2   = 0.0;   // sum of squared deviations from the running mean

    void update(double x) {
        n++;
        const double delta = x - mean;
        mean += delta / n;
        m2   += delta * (x - mean);
    }

    // sample variance (n-1). returns 0 if fewer than 2 samples.
    double variance() const { return n > 1 ? m2 / (n - 1) : 0.0; }
    double stddev()   const { return std::sqrt(variance()); }
};

// per-layer accumulated statistics
struct layer_data_t {
    // stats against previous layer
    running_stats_t cos_sim;  // cos(h_N, h_{N-1}) per token
    running_stats_t l2_dist;  // ||h_N - h_{N-1}||_2 per token

    // stats of this layer
    running_stats_t l2_norm;  // ||h_N||_2 per token
    running_stats_t l1_norm;  // ||h_N||_1 per token
    running_stats_t inf_norm; // ||h_N||_inf per token
};

struct cb_data_t {
    int n_layer = 0; // number of layers in the model (for bounds / sanity checks)

    // per-layer running stats, indexed by layer number
    std::map<int, layer_data_t> stats;

    // previous layer's hidden state, per token: prev[layer] holds h_{layer-1}
    // laid out as [n_embd] contiguous floats per token
    std::map<int, std::vector<float>> prev;
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
        return 0.0f;
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

// helper struct to access tensor data regardless of backend
struct tensor_data_view {
    tensor_data_view(void * ptr) : ptr_(ptr), buf_({}) {}
    tensor_data_view(std::vector<uint8_t>&& buf) : ptr_(buf.data()), buf_(std::move(buf)) {}
    inline void * data() const { return ptr_; }
    private:
        void * ptr_;
        std::vector<uint8_t> buf_;
};

static tensor_data_view tensor_get_view(const ggml_tensor * t) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        return tensor_data_view(t->data);
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    return tensor_data_view(std::move(buf));
}

static std::vector<float> tensor_get_data_f32(const ggml_tensor * t) {
    GGML_ASSERT(t != nullptr);

    const int64_t n_elem = ggml_nelements(t);
    std::vector<float> out(n_elem);

    const tensor_data_view view = tensor_get_view(t);
    const void * t_data = view.data();

    switch (t->type) {
        case GGML_TYPE_F32: {
            const float * src = static_cast<const float *>(t_data);
            std::copy(src, src + n_elem, out.begin());
        } break;
        case GGML_TYPE_F16: {
            const ggml_fp16_t * src = static_cast<const ggml_fp16_t *>(t_data);
            for (int64_t i = 0; i < n_elem; ++i) {
                out[i] = ggml_fp16_to_fp32(src[i]);
            }
        } break;
        case GGML_TYPE_BF16: {
            const ggml_bf16_t * src = static_cast<const ggml_bf16_t *>(t_data);
            for (int64_t i = 0; i < n_elem; ++i) {
                out[i] = ggml_bf16_to_fp32(src[i]);
            }
        } break;
        default: GGML_ABORT("%s: unhandled tensor type %s", __func__, ggml_type_name(t->type));
    }

    return out;
}

// filter: only ask for the layer-output node.
// TODO: extend match set to "l_last" to support mHC models
static bool is_layer_output(const ggml_tensor * t) {
    if (t == nullptr || t->name[0] == '\0') {
        return false;
    }
    const char * prefix = "l_out-";
    const size_t plen   = strlen(prefix);
    if (strncmp(t->name, prefix, plen) != 0) {
        return false;
    }
    return t->name[plen] != '\0' && isdigit(static_cast<unsigned char>(t->name[plen]));
}

// parse the layer index out of a `l_out-<il>` node name.
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

    // `l_out` shape is [n_embd, n_tokens, ...]
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    // non-const so we can move it into `prev` at the end (avoids a full copy)
    std::vector<float> data_vec = tensor_get_data_f32(t);
    if (data_vec.empty()) {
        return true;
    }
    const float * data = data_vec.data();

    auto & st = state->stats[layer];

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        const float * h = data + tok * n_embd;

        // self metrics (always computable)
        st.l2_norm .update(l2_norm (h, n_embd));
        st.l1_norm .update(l1_norm (h, n_embd));
        st.inf_norm.update(inf_norm(h, n_embd));

        // pairwise metrics (need the previous layer's vector for this token).
        // NOTE: within one llama_decode() the layers run in topological order,
        // so prev[layer-1] always holds the current batch's data by the time
        // we get here.
        const float * prev = nullptr;
        if (layer > 0) {
            auto it = state->prev.find(layer - 1);
            if (it != state->prev.end() &&
                static_cast<int64_t>(it->second.size()) == n_embd * n_tokens) {
                prev = it->second.data() + tok * n_embd;
            }
        }

        if (prev) {
            st.cos_sim.update(cos_sim(h, prev, n_embd));
            st.l2_dist.update(l2_dist(h, prev, n_embd));
        }
    }

    // move, don't copy
    state->prev[layer] = std::move(data_vec);

    return true;
}

//
// reporting
//

struct layer_row {
    int    layer;
    double cos_sim,    cos_sim_std;
    double l2_dist,    l2_dist_std;
    double l2,         l2_std;
    double l1,         l1_std;
    double linf,       linf_std;
};

// center `text` (assumed ASCII) over `w` bytes; pads with spaces
static void print_centered(const char * text, int w) {
    const int len = (int)strlen(text);
    const int pad = std::max(0, w - len);
    const int left  = pad / 2;
    const int right = pad - left;
    for (int i = 0; i < left;  ++i) putchar(' ');
    fputs(text, stdout);
    for (int i = 0; i < right; ++i) putchar(' ');
}

static void print_report(const cb_data_t & state) {
    std::vector<layer_row> rows;

    for (const auto & kv : state.stats) {
        const int layer = kv.first;
        const auto & s  = kv.second;
        if (s.l2_norm.n <= 0) {
            continue;
        }
        rows.push_back(layer_row{
            layer,
            s.cos_sim .mean, s.cos_sim .stddev(),
            s.l2_dist .mean, s.l2_dist .stddev(),
            s.l2_norm .mean, s.l2_norm .stddev(),
            s.l1_norm .mean, s.l1_norm .stddev(),
            s.inf_norm.mean, s.inf_norm.stddev(),
        });
    }

    if (g_rank_metric == 1) {
        std::sort(rows.begin(), rows.end(), [](const layer_row & a, const layer_row & b) {
            return a.cos_sim < b.cos_sim;
        });
    } else {
        std::sort(rows.begin(), rows.end(), [](const layer_row & a, const layer_row & b) {
            return a.l2_dist > b.l2_dist;
        });
    }

    // --- column geometry -----------------------------------------------------
    //
    // Each metric cell is built as:  <mean> <space><±><space> <std>
    //
    //   W_MEAN = 8   -> %8.3f fits up to "9999.999" (room for 2313.284)
    //   W_STD  = 6   -> %6.2f  fits up to  "999.99" (room for  490.72)
    //                   and guarantees at least 2 whole-number digits of pad
    //   " ± "        -> 4 bytes in UTF-8 (space=1, ±=2, space=1)
    //
    const int W_RANK  = 4;
    const int W_LAYER = 5;
    const int W_MEAN  = 8;
    const int W_STD   = 6;
    const int W_SEP   = 4;                              // " ± " as UTF-8
    const int W_MET   = W_MEAN + W_SEP + W_STD;         // = 18 bytes/cell

    // --- header (leading space on the line) --------------------------------
    printf("\n\n %*s  %*s  ", W_RANK, "rank", W_LAYER, "layer");
    print_centered("cossim",        W_MET); printf("  ");
    print_centered("L2 norm dist.", W_MET); printf("  ");
    print_centered("L2 norm",       W_MET); printf("  ");
    print_centered("L1 norm",       W_MET); printf("  ");
    print_centered("L-inf norm",    W_MET);
    printf("\n");

    // --- rule ---------------------------------------------------------------
    // 1 (leading space) + rank + 2 + layer + 2 + 5 cells + 4 inter-cell gaps
    const int total_w = 1 + W_RANK + 2 + W_LAYER + 2 + 5 * W_MET + 4 * 2;
    printf(" %s\n", std::string(total_w - 1, '=').c_str());

    // --- data rows (leading space on each line) ----------------------------
    int rank = 1;
    for (const auto & r : rows) {
        char cos_buf [64], l2d_buf[64], l2_buf[64], l1_buf[64], linf_buf[64];

        // %*.*f -> width, precision taken from args; gives fixed byte width
        std::snprintf(cos_buf,  sizeof(cos_buf),
            "%*.*f ± %*.*f", W_MEAN, 3, r.cos_sim,  W_STD, 2, r.cos_sim_std);
        std::snprintf(l2d_buf,  sizeof(l2d_buf),
            "%*.*f ± %*.*f", W_MEAN, 3, r.l2_dist,  W_STD, 2, r.l2_dist_std);
        std::snprintf(l2_buf,   sizeof(l2_buf),
            "%*.*f ± %*.*f", W_MEAN, 3, r.l2,       W_STD, 2, r.l2_std);
        std::snprintf(l1_buf,   sizeof(l1_buf),
            "%*.*f ± %*.*f", W_MEAN, 3, r.l1,       W_STD, 2, r.l1_std);
        std::snprintf(linf_buf, sizeof(linf_buf),
            "%*.*f ± %*.*f", W_MEAN, 3, r.linf,     W_STD, 2, r.linf_std);

        printf(" %*d  %*d  %s  %s  %s  %s  %s\n",
               W_RANK,  rank++,
               W_LAYER, r.layer,
               cos_buf, l2d_buf, l2_buf, l1_buf, linf_buf);
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

    const int n_ctx   = static_cast<int>(llama_n_ctx(ctx));
    const int n_batch = static_cast<int>(llama_n_batch(ctx));

    // truncate to n_ctx (NOT n_batch) -- we'll iterate over batches below
    if (static_cast<int>(tokens.size()) > n_ctx) {
        LOG_WRN("%s: truncating prompt from %zu to %d tokens (n_ctx)\n",
                __func__, tokens.size(), n_ctx);
        tokens.resize(n_ctx);
    }

    const int n_tokens  = static_cast<int>(tokens.size());
    const int n_batches = (n_tokens + n_batch - 1) / n_batch;

    LOG_INF("%s: decoding %d tokens in %d batch(es) of up to %d tokens\n",
            __func__, n_tokens, n_batches, n_batch);

    for (int b = 0; b < n_batches; ++b) {
        const int start = b * n_batch;
        const int n     = std::min(n_batch, n_tokens - start);

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = tokens[start + i];
            batch.pos[i]       = start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = false;
        }
        batch.n_tokens = n;

        LOG_INF("%s: batch %d/%d: %d tokens (positions %d..%d)\n",
                __func__, b + 1, n_batches, n, start, start + n - 1);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed on batch %d\n", __func__, b + 1);
            llama_batch_free(batch);
            return;
        }

        llama_batch_free(batch);
    }
}

static void print_usage(int /*argc*/, char ** /*argv*/) {
    printf("\nllama-layer-stats: per-layer hidden-state statistics\n\n");
    printf("Additional options (beyond common llama.cpp options):\n");
    printf("  --rank-by <metric>   metric used to rank layers in the report\n");
    printf("                       l2dist  - L2 distance to previous layer (default)\n");
    printf("                       cossim  - cosine similarity to previous layer\n\n");
}

// Pre-scan argv, extract `--rank-by <value>`, and remove it from argv so
// common_params_parse() doesn't reject it as an unknown flag.
static void strip_rank_metric_arg(int & argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rank-by") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --rank-by requires an argument (l2dist or cossim)\n");
                exit(1);
            }
            if (strcmp(argv[i + 1], "l2dist") == 0) {
                g_rank_metric = 0;
            } else if (strcmp(argv[i + 1], "cossim") == 0) {
                g_rank_metric = 1;
            } else {
                fprintf(stderr, "error: unknown --rank-by value '%s' (use l2dist or cossim)\n",
                        argv[i + 1]);
                exit(1);
            }
            // shift remaining args left by 2
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--; // re-check the same index
        }
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // must happen before common_params_parse()
    strip_rank_metric_arg(argc, argv);

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

    state.n_layer = llama_model_n_layer(model);

    run_layer_stats(ctx, params);

    print_report(state);

    llama_backend_free();
    return 0;
}

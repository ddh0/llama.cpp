/**
 *
 * llama-layer-stats -- record per-layer hidden state statistics captured via eval callback
 *
 * TODO: support `l_last` for mHC models (3-dimensional hidden states)
**/

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <cctype>
#include <limits>
#include <map>
#include <string>
#include <vector>

// --rank-by CLI options
//
//   0 = rank by L2 distance to previous layer, descending (most change first)
//   1 = rank by cosine similarity to previous layer, ascending (most change first)
//   2 = rank by composite importance score, descending (most important first) (default)
static int g_rank_metric = 2;

// Welford's online algorithm for running mean and variance
// ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
struct running_stats_t {
    int64_t n    = 0;   // number of observations
    double  mean = 0.0; // running mean
    double  m2   = 0.0; // sum of squared deviations from the running mean

    void update(double x) {
        n++;
        const double delta = x - mean;
        mean += delta / n;
        m2   += delta * (x - mean);
    }

    double variance() const { return n > 1 ? m2 / (n - 1) : 0.0; }
    double stddev()   const { return std::sqrt(variance()); }
};

// per-layer statistics, accumulated per token
struct layer_data_t {
    // these compare to previous layer
    running_stats_t cos_sim;  // cos(h_N, h_{N-1})
    running_stats_t l2_dist;  // ||h_N - h_{N-1}||_2

    // these only consider the current layer
    running_stats_t l2_norm;  // ||h_N||_2
    running_stats_t l1_norm;  // ||h_N||_1
    running_stats_t inf_norm; // ||h_N||_inf
};

struct cb_data_t {
    int n_layer = 0;                        // current layer idx
    int n_tokens = 0;                       // # of tokens seen
    size_t n_bytes = 0;                     // total size of all observed hidden states, in bytes
    std::map<int, layer_data_t> stats;      // per-layer running stats, indexed by layer number
    std::map<int, std::vector<float>> prev; // prev. layer hidden state: prev[layer] -> h_{layer-1}
                                            // laid out as [n_embd] contiguous floats per token
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
// TODO: extend for mHC
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

// get layer index from tensor name `l_out-<il>`
static int tensor_get_layer_idx(const ggml_tensor * t) {
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

static bool cb_func(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = static_cast<cb_data_t *>(user_data);

    if (ask) {
        return is_layer_output(t);
    }

    const int il = tensor_get_layer_idx(t);
    if (il < 0) {
        LOG_ERR("%s: could not parse layer index from tensor name '%s'\n", __func__, t->name);
        return true;
    }
    if (data->n_layer > 0 && il >= data->n_layer) {
        LOG_ERR("%s: layer index %d out of range (n_layer=%d)\n", __func__, il, data->n_layer);
        return true;
    }

    // shape of `l_out` is [n_embd, n_tokens, <ne2>, <ne3>]
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    // non-const, avoid full copy
    std::vector<float> hidden_vec_f32 = tensor_get_data_f32(t);
    if (hidden_vec_f32.empty()) {
        return true;
    }
    const float * hidden_f32 = hidden_vec_f32.data();

    data->n_bytes += n_embd * n_tokens * sizeof(float);

    auto & st = data->stats[il];

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        const float * h = hidden_f32 + tok * n_embd;

        // basic stats
        st.l2_norm .update(l2_norm (h, n_embd));
        st.l1_norm .update(l1_norm (h, n_embd));
        st.inf_norm.update(inf_norm(h, n_embd));

        // pairwise metrics (against previous layer for the same token)
        const float * prev = nullptr;
        if (il > 0) {
            auto it = data->prev.find(il - 1);
            if (it != data->prev.end() &&
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
    data->prev[il] = std::move(hidden_vec_f32);

    return true;
}

//
// reporting
//

// global CLI option: path to write per-layer stats as JSON (empty = no export)
static std::string g_output_path;

struct layer_row {
    int          layer;
    layer_data_t stats;
    double       score = -1.0; // composite importance score; -1 = not available
};

// compute normalized importance score, as a composition of cos_sim and l2_dist:
//
//   x = max(1 - norm(cos_sim) + eps, eps)  (high when cos_sim is low)
//   y = max(    norm(l2_dist) + eps, eps)  (high when l2_dist is high)
//
//   score = x^a + x * y^(1 - a)
//
// where norm(v) = (v - min v) / (max v - min v) over all layers where both stats exist
//
static void compute_composite_scores(std::vector<layer_row> & rows) {
    const double eps = 1e-5;
    const double a   = 0.2;

    double c_min =  std::numeric_limits<double>::max();
    double c_max = -std::numeric_limits<double>::max();
    double d_min =  std::numeric_limits<double>::max();
    double d_max = -std::numeric_limits<double>::max();
    int    n_valid = 0;

    for (const auto & r : rows) {
        if (r.stats.cos_sim.n <= 0 || r.stats.l2_dist.n <= 0) {
            continue;
        }
        c_min = std::min(c_min, r.stats.cos_sim.mean);
        c_max = std::max(c_max, r.stats.cos_sim.mean);
        d_min = std::min(d_min, r.stats.l2_dist.mean);
        d_max = std::max(d_max, r.stats.l2_dist.mean);
        n_valid++;
    }

    if (n_valid == 0) {
        return; // nothing to score
    }

    const double c_span = (c_max > c_min) ? (c_max - c_min) : 1.0;
    const double d_span = (d_max > d_min) ? (d_max - d_min) : 1.0;

    for (auto & r : rows) {
        if (r.stats.cos_sim.n <= 0 || r.stats.l2_dist.n <= 0) {
            r.score = -1.0;
            continue;
        }
        const double cN = (r.stats.cos_sim.mean - c_min) / c_span; // 0..1
        const double dN = (r.stats.l2_dist.mean - d_min) / d_span; // 0..1
        const double x  = std::max(1.0 - cN + eps, eps);
        const double y  = std::max(      dN + eps, eps);
        r.score = std::pow(x, a) + x * std::pow(y, (1.0f - a));
    }
}

// center `text` (assumed ASCII) over `w` bytes; pad with spaces
static void print_centered(const char * text, int w) {
    const int len = (int)strlen(text);
    const int pad = std::max(0, w - len);
    const int left  = pad / 2;
    const int right = pad - left;
    for (int i = 0; i < left;  ++i) putchar(' ');
    fputs(text, stdout);
    for (int i = 0; i < right; ++i) putchar(' ');
}

static void print_report(const cb_data_t & data) {
    std::vector<layer_row> rows;
    std::vector<int>       empty_layers;

    for (int layer = 0; layer < data.n_layer; ++layer) {
        const auto it = data.stats.find(layer);
        if (it == data.stats.end() || it->second.l2_norm.n <= 0) {
            empty_layers.push_back(layer);
            continue;
        }
        rows.push_back(layer_row{layer, it->second, -1.0});
    }

    compute_composite_scores(rows);

    if (g_rank_metric == 2) {
        std::sort(rows.begin(), rows.end(), [](const layer_row & a, const layer_row & b) {
            return a.score > b.score; // most important first
        });
    } else if (g_rank_metric == 1) {
        std::sort(rows.begin(), rows.end(), [](const layer_row & a, const layer_row & b) {
            return a.stats.cos_sim.mean < b.stats.cos_sim.mean;
        });
    } else {
        std::sort(rows.begin(), rows.end(), [](const layer_row & a, const layer_row & b) {
            return a.stats.l2_dist.mean > b.stats.l2_dist.mean;
        });
    }

    const int W_RANK      = 4;
    const int W_LAYER     = 5;
    const int W_COMPOSITE = 12;
    const int W_MEAN      = 9;
    const int W_STD       = 7;
    const int W_SEP       = 4; // " ± "
    const int W_METRIC    = W_MEAN + W_SEP + W_STD;

    auto format_stat = [&](char * buf, size_t buflen, const running_stats_t & s) {
        if (s.n <= 0) {
            std::snprintf(buf, buflen, "%*s", W_METRIC - 1, "no data    ");
        } else if (s.n <= 1) {
            std::snprintf(buf, buflen, "%*.*f", W_MEAN, 3, s.mean);
        } else {
            std::snprintf(buf, buflen, "%*.*f ± %*.*f", W_MEAN, 3, s.mean, W_STD, 2, s.stddev());
        }
    };

    fflush(stdout);
    printf("\n\n  %*s  %*s  ", W_RANK, "rank", W_LAYER, "layer");
    print_centered("composite",     W_COMPOSITE);   printf("  ");
    print_centered("cossim",        W_METRIC - 1);  printf("  ");
    print_centered("L2 norm dist.", W_METRIC - 1);  printf("  ");
    print_centered("L2 norm",       W_METRIC - 1);  printf("  ");
    print_centered("L1 norm",       W_METRIC - 1);  printf("  ");
    print_centered("L-inf norm",    W_METRIC - 1);
    printf("\n");

    // rule
    const int total_w = 3 + W_RANK + 2 + W_LAYER + 2 + W_COMPOSITE + 2
                      + 5 * (W_METRIC - 1) + 4 * 2;
    printf("%s\n", std::string(total_w, '=').c_str());

    auto emit = [&](const char * rank_str, int layer, const layer_data_t & s, double score) {
        char cos_buf[64], l2d_buf[64], l2_buf[64], l1_buf[64], linf_buf[64];
        char score_buf[32];
        format_stat(cos_buf,  sizeof(cos_buf),  s.cos_sim);
        format_stat(l2d_buf,  sizeof(l2d_buf),  s.l2_dist);
        format_stat(l2_buf,   sizeof(l2_buf),   s.l2_norm);
        format_stat(l1_buf,   sizeof(l1_buf),   s.l1_norm);
        format_stat(linf_buf, sizeof(linf_buf), s.inf_norm);
        if (score >= 0.0) {
            std::snprintf(score_buf, sizeof(score_buf), "%*.*f", W_COMPOSITE, 4, score);
        } else {
            std::snprintf(score_buf, sizeof(score_buf), "%*s", W_COMPOSITE, "-");
        }

        printf(" %*s  %*d  %s  %s  %s  %s  %s  %s\n",
               W_RANK, rank_str, W_LAYER, layer,
               score_buf,
               cos_buf, l2d_buf, l2_buf, l1_buf, linf_buf);
    };

    int rank = 1;
    for (const auto & r : rows) {
        char rank_buf[16];
        std::snprintf(rank_buf, sizeof(rank_buf), "%d", rank++);
        emit(rank_buf, r.layer, r.stats, r.score);
    }

    const layer_data_t empty;
    for (int layer : empty_layers) {
        emit("-", layer, empty, -1.0);
    }

    printf("\n\n");
    fflush(stdout);

    LOG_INF("%s: computed over %d tokens, %d layers\n", __func__, data.n_tokens, data.n_layer);
    LOG_INF("%s: total hidden state data captured: %.2f MiB\n", __func__, data.n_bytes/1024.0/1024.0);
}

// mean or stddev of a running stat, or null
static common_json_value stat_json(const running_stats_t & s, bool want_std) {
    if (s.n <= 0 || (want_std && s.n <= 1)) {
        return nullptr; // null
    }
    return want_std ? s.stddev() : s.mean;
}

// write per-layer stats to a JSON file
static bool write_json(const cb_data_t & state, const std::string & path) {
    if (path.empty()) {
        return true;
    }

    common_json root = common_json::object();
    root["layers"] = common_json::array();

    const layer_data_t empty; // all running_stats_t have n == 0
    for (int layer = 0; layer < state.n_layer; ++layer) {
        const auto it = state.stats.find(layer);
        const layer_data_t & s = (it != state.stats.end()) ? it->second : empty;

        root["layers"].push_back({
            {"layer",        layer},
            {"cos_sim",      stat_json(s.cos_sim,  false)},
            {"cos_sim_std",  stat_json(s.cos_sim,  true)},
            {"l2_dist",      stat_json(s.l2_dist,  false)},
            {"l2_dist_std",  stat_json(s.l2_dist,  true)},
            {"l2_norm",      stat_json(s.l2_norm,  false)},
            {"l2_norm_std",  stat_json(s.l2_norm,  true)},
            {"l1_norm",      stat_json(s.l1_norm,  false)},
            {"l1_norm_std",  stat_json(s.l1_norm,  true)},
            {"inf_norm",     stat_json(s.inf_norm, false)},
            {"inf_norm_std", stat_json(s.inf_norm, true)},
        });
    }

    const std::string text = root.dump(2);

    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, path.c_str());
        return false;
    }
    fwrite(text.data(), 1, text.size(), f);
    fclose(f);

    LOG_INF("%s: saved statistics to '%s'\n", __func__, path.c_str());
    return true;
}

// eval the input, return true on success, false on failure
static bool run_layer_stats(llama_context * ctx, const common_params & params, cb_data_t & data) {
    LOG_INF("%s: tokenizing the input ...\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);
    if (tokens.empty()) {
        LOG_ERR("%s: input is empty!\n", __func__);
        return false;
    }

    const int n_ctx   = static_cast<int>(llama_n_ctx(ctx));
    const int n_batch = static_cast<int>(llama_n_batch(ctx));

    // truncate to n_ctx
    if (static_cast<int>(tokens.size()) > n_ctx) {
        LOG_WRN("%s: truncating the input from %zu to n_ctx == %d tokens\n",
                __func__, tokens.size(), n_ctx);
        tokens.resize(n_ctx);
    }

    const int n_tokens = static_cast<int>(tokens.size());
    const int n_batches = (n_tokens + n_batch - 1) / n_batch;
    data.n_tokens = n_tokens;

    LOG_INF("%s: decoding %d tokens in %d %s\n",
            __func__, n_tokens, n_batches, n_batches == 1 ? "batch" : "batches");

    for (int b = 0; b < n_batches; ++b) {
        const int start = b * n_batch;
        const int n     = std::min(n_batch, n_tokens - start);

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = tokens[start + i];
            batch.pos[i]       = start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = true; // ensure crop_before_nextn() -> false
        }
        batch.n_tokens = n;

        LOG_INF("%s: batch %d/%d: decoding %d tokens ...\n", __func__, b + 1, n_batches, n);

        auto ret = llama_decode(ctx, batch);
        if (ret != 0) {
            LOG_ERR("%s: llama_decode failed on batch %d (code %d)\n", __func__, b + 1, ret);
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);
    }

    return true;
}

static void print_usage(int /*argc*/, char ** /*argv*/) {
    printf("\nllama-layer-stats: record per-layer hidden state statistics captured via eval callback\n\n");
    printf("options:\n\n");
    printf("  --rank-by METRIC     metric used to rank layers in the report. one of:\n");
    printf("                         composite - combined cos_sim and l2_dist (default)\n");
    printf("                         cos_sim   - cosine similarity to previous layer\n");
    printf("                         l2_dist   - L2 distance to previous layer\n");
    printf("  --output-json PATH   write per-layer stats to a JSON file\n\n");
}

// pre-scan argv, extract custom flags (`--rank-by`, `--output-json`), and remove
// them from argv so common_params_parse() doesn't reject them as unknown flags.
static void strip_custom_args(int & argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rank-by") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --rank-by requires an argument (composite, l2_dist or cos_sim)\n");
                exit(1);
            }
            if (strcmp(argv[i + 1], "composite") == 0) {
                g_rank_metric = 2;
            } else if (strcmp(argv[i + 1], "l2_dist") == 0) {
                g_rank_metric = 0;
            } else if (strcmp(argv[i + 1], "cos_sim") == 0) {
                g_rank_metric = 1;
            } else {
                fprintf(stderr, "error: unknown --rank-by value '%s' (use composite, l2_dist or cos_sim)\n",
                        argv[i + 1]);
                exit(1);
            }
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;
        } else if (strcmp(argv[i], "--output-json") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --output-json requires an argument (path)\n");
                exit(1);
            }
            g_output_path = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;
        }
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C"); // ref: https://github.com/ggml-org/llama.cpp/pull/17331

    // must happen before common_params_parse() // TODO: properly integrate into common args
    strip_custom_args(argc, argv);

    common_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LAYER_STATS, print_usage)) {
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("no input provided; please specify -p / --prompt OR -f / --file\n");
        return 1;
    }

    cb_data_t cb_data;

    params.cb_eval = cb_func;
    params.cb_eval_user_data = &cb_data;

    llama_backend_init();
    llama_numa_init(params.numa);

    if (params.warmup) {
        LOG_WRN("override: set warmup = false\n");
        params.warmup = false; // avoid capturing an empty warmup pass
    }
    if (params.n_parallel > 1) {
        LOG_WRN("override: set n_parallel = 1\n");
        params.n_parallel = 1; // NOTE: only support one sequence for now
    }

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init llama_model\n", __func__);
        return 1;
    }

    if (ctx == nullptr) {
        LOG_ERR("%s: failed to init llama_context\n", __func__);
        return 1;
    }

    cb_data.n_layer = llama_model_n_layer(model);

    run_layer_stats(ctx, params, cb_data);

    print_report(cb_data);

    if (!write_json(cb_data, g_output_path)) {
        LOG_ERR("%s: failed to write JSON output\n", __func__);
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}

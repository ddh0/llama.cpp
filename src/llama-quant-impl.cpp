// TODO: definitely don't need all of these includes
#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <cmath>
#include <cstring>
#include <string>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                LLAMA_LOG_DEBUG("(blk.%d imatrix) ", p.first);
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

// based on the tensor name, is it a token embeddings tensor?
static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

// based on the tensor name, is it an output tensor?
static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

// tensor categorization for quantization
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

// quantization state tracking
struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tensor_types) {
            const auto & tensor_types = *static_cast<const std::vector<tensor_type_option> *>(params->tensor_types);
            for (const auto & [tname, qtype] : tensor_types) {
                tensor_type_patterns.emplace_back(std::regex(tname), qtype);
            }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};

// dequantize tensor
static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

// do we allow this tensor to be quantized?
// TODO: make this return as early as possible to avoid unnecessary string checks (use string view)
static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first -- no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string name = ggml_get_name(tensor);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd.") == std::string::npos;

    return quantize;
}

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs.n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

// quantize tensor
static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

// do we require importance matrix data for this tensor?
static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

// given a file type, get the default tensor type
static ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }
}
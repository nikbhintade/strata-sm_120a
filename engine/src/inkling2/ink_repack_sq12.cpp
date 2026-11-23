// engine/src/inkling2/ink_repack_sq12.cpp - Multithreaded .strata v1.1 SQ-15 -> SQ-12 (128K full-VRAM 16 GB) repacker
// Keeps the full 2.07 GB SQ-15 trunk (embed, unembed, attention, controllers, DSpark) verbatim and repacks
// SF4 (and optionally L20..23 S4_K) MoE experts to MSE-optimal 3-bit S3_K with zero 2-bit S2_K matrices!
// MIT License - Copyright (c) 2026 Strata contributors

#include "ink_gguf_transcode.h"
#include "ink_manifest.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::artifact::STRATA_NUM_SECTIONS;
using strata::artifact::STRATA_PAGE_SIZE;
using strata::artifact::StrataExpertEntry;
using strata::artifact::StrataFileReader;
using strata::artifact::StrataSectionEntry;
using strata::quant::sk::SkCode;

inline uint64_t align_up(uint64_t x, uint64_t a) noexcept {
    return (x + a - 1u) & ~(a - 1u);
}

bool seek_abs(std::FILE* fp, uint64_t off) noexcept {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

bool write_all(std::FILE* fp, const void* data, size_t bytes) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (bytes > 0) {
        size_t n = std::fwrite(p, 1, bytes, fp);
        if (n == 0) return false;
        p += n;
        bytes -= n;
    }
    return true;
}

bool pad_to_page(std::FILE* fp, uint64_t& cur_off) noexcept {
    const uint64_t aligned = align_up(cur_off, STRATA_PAGE_SIZE);
    if (aligned == cur_off) return true;
    static const uint8_t zeros[STRATA_PAGE_SIZE]{};
    const size_t pad = static_cast<size_t>(aligned - cur_off);
    if (!write_all(fp, zeros, pad)) return false;
    cur_off = aligned;
    return true;
}

// MSE-optimal 3-bit S3_K super-block encoder (256 weights = 4 x 64-weight sub-groups -> 102 bytes).
// Bit-exact wire format matching strata::quant::sk::dequant_sk_row(SkCode::S3_K) & CUDA k_ink_moe_sk_tc_fmt<S3_K>.
void encode_s3k_superblock_mse(const float* w, uint8_t* dst) noexcept {
    static constexpr float kAlphas[] = {0.78f, 0.84f, 0.89f, 0.93f, 0.96f, 0.985f, 1.00f};
    float s_opt[4]{};
    float d_max = 0.0f;

    for (int g = 0; g < 4; ++g) {
        const float* wg = w + g * 64;
        float amax = 0.0f;
        for (int j = 0; j < 64; ++j) {
            const float a = std::fabs(wg[j]);
            if (a > amax) amax = a;
        }
        if (!(amax > 0.0f)) {
            s_opt[g] = 0.0f;
            continue;
        }
        const float base_s = amax * (1.0f / 3.5f);
        float best_s = base_s;
        float best_err = 1e30f;
        for (float alpha : kAlphas) {
            const float sg = base_s * alpha;
            const float inv = 1.0f / sg;
            float err = 0.0f;
            for (int j = 0; j < 64; ++j) {
                float qf = std::nearbyint(wg[j] * inv + 3.5f);
                if (qf < 0.0f) qf = 0.0f;
                else if (qf > 7.0f) qf = 7.0f;
                const float diff = wg[j] - (qf - 3.5f) * sg;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_s = sg;
            }
        }
        s_opt[g] = best_s;
        if (best_s > d_max) d_max = best_s;
    }

    const uint16_t d_bits = ink2::transcode::f32_to_fp16_bits(d_max);
    const float D = ink2::transcode::fp16_bits_to_f32(d_bits);
    if (!(D > 0.0f)) {
        std::memset(dst, 0, 102);
        return;
    }

    std::memcpy(dst, &d_bits, 2);
    uint8_t* u = dst + 2;
    uint8_t* p_lo = dst + 6;
    uint8_t* p_hi = dst + 70;
    const float inv_D_255 = 255.0f / D;

    uint8_t qbuf[256];
    for (int g = 0; g < 4; ++g) {
        int ug = static_cast<int>(std::lround(s_opt[g] * inv_D_255));
        if (s_opt[g] > 0.0f && ug < 1) ug = 1;
        if (ug > 255) ug = 255;
        u[g] = static_cast<uint8_t>(ug);
        const float eff = D * (static_cast<float>(ug) * (1.0f / 255.0f));
        const float inv = (eff > 0.0f) ? (1.0f / eff) : 0.0f;
        const float* wg = w + g * 64;
        uint8_t* qg = qbuf + g * 64;
        for (int j = 0; j < 64; ++j) {
            int q = static_cast<int>(std::lround(wg[j] * inv + 3.5f));
            if (q < 0) q = 0;
            else if (q > 7) q = 7;
            qg[j] = static_cast<uint8_t>(q);
        }
    }

    for (int i = 0; i < 64; ++i) {
        p_lo[i] = static_cast<uint8_t>(
            (qbuf[4 * i + 0] & 0x03u) |
            ((qbuf[4 * i + 1] & 0x03u) << 2) |
            ((qbuf[4 * i + 2] & 0x03u) << 4) |
            ((qbuf[4 * i + 3] & 0x03u) << 6));
    }
    for (int i = 0; i < 32; ++i) {
        uint8_t h = 0;
        for (int b = 0; b < 8; ++b) {
            h |= static_cast<uint8_t>(((qbuf[8 * i + b] >> 2) & 1u) << b);
        }
        p_hi[i] = h;
    }
}

bool selfcheck_s3k_codec() {
    float src[256];
    for (int g = 0; g < 4; ++g) {
        const float amp = (g == 0) ? 0.05f : (g == 1) ? 0.12f : (g == 2) ? 0.02f : 0.25f;
        for (int j = 0; j < 64; ++j) {
            src[g * 64 + j] = amp * std::sin(static_cast<float>(g * 64 + j) * 0.37f + 0.1f);
        }
    }
    uint8_t enc[102]{};
    encode_s3k_superblock_mse(src, enc);
    float dec[256]{};
    strata::quant::sk::dequant_sk_row(enc, SkCode::S3_K, dec, 256);
    float max_rel = 0.0f;
    for (int g = 0; g < 4; ++g) {
        float amax = 0.0f, err = 0.0f;
        for (int j = 0; j < 64; ++j) {
            amax = std::max(amax, std::fabs(src[g * 64 + j]));
            err = std::max(err, std::fabs(src[g * 64 + j] - dec[g * 64 + j]));
        }
        max_rel = std::max(max_rel, err / std::max(amax, 1e-8f));
    }
    if (max_rel > 0.25f) {
        std::fprintf(stderr, "[repack] FATAL: S3_K codec selfcheck failed (max_rel=%.4f)\n", max_rel);
        return false;
    }
    return true;
}

void repack_one_expert(
    const uint8_t* in_body,
    const StrataExpertEntry& in_ent,
    size_t hidden,
    size_t inter,
    bool keep_coda,
    int phys_layer,
    std::vector<uint8_t>& out_body,
    StrataExpertEntry& out_ent) {
    out_ent = in_ent;
    const auto gu = static_cast<SkCode>(in_ent.gu_code_id);
    const auto dn = static_cast<SkCode>(in_ent.dn_code_id);

    const bool convert_sf4 = (gu == SkCode::SF4 && dn == SkCode::SF4);
    const bool convert_s4k = (!keep_coda && gu == SkCode::S4_K && dn == SkCode::S4_K);
    (void)phys_layer;

    if (!convert_sf4 && !convert_s4k) {
        out_body.assign(in_body, in_body + in_ent.body_bytes);
        return;
    }

    uint32_t new_dn_off = 0, new_body_bytes = 0;
    ink2::transcode::transcoded_sk_expert_layout(
        SkCode::S3_K, SkCode::S3_K, hidden, inter, new_dn_off, new_body_bytes);
    out_body.assign(new_body_bytes, 0);

    const size_t sb_s3 = strata::quant::sk::superblock_bytes(SkCode::S3_K);  // 102
    const size_t n_sb_gu = hidden / 256u;                                    // 9
    const size_t n_sb_dn = (2u * inter) / 256u;                              // 9
    const size_t row_b_s3_gu = n_sb_gu * sb_s3;
    const size_t pair_b_s3_dn = n_sb_dn * sb_s3;

    std::vector<float> row_f32(hidden);
    std::vector<float> pair_f32(2u * inter);

    uint8_t* dst_gu = out_body.data() + 2u * inter;
    uint8_t* dst_dn = out_body.data() + new_dn_off;

    if (convert_sf4) {
        // SF4 has no MACaQ pre-scaling -> identity sq_xd_dn = FP16 1.0f (0x3C00)
        const uint16_t one_f16 = 0x3C00u;
        for (size_t j = 0; j < inter; ++j) {
            std::memcpy(out_body.data() + 2u * j, &one_f16, 2);
        }
        float hdr[4];
        std::memcpy(hdr, in_body, 16);
        const float w13_outer = hdr[0] / hdr[1];
        const float w2_outer  = hdr[2] / hdr[3];

        const uint8_t* w13_p = in_body + 16;
        const uint8_t* w13_s = w13_p + (2u * inter) * (hidden / 2u);
        for (size_t r = 0; r < 2u * inter; ++r) {
            strata::quant::sk::dequant_sf4_row(
                w13_p + r * (hidden / 2u), w13_s + r * (hidden / 16u), w13_outer, row_f32.data(), hidden);
            uint8_t* dr = dst_gu + r * row_b_s3_gu;
            for (size_t s = 0; s < n_sb_gu; ++s) {
                encode_s3k_superblock_mse(row_f32.data() + s * 256u, dr + s * sb_s3);
            }
        }

        const uint8_t* w2_p = in_body + in_ent.dn_offset;
        const uint8_t* w2_s = w2_p + hidden * (inter / 2u);
        for (size_t p = 0; p < hidden / 2u; ++p) {
            const size_t r0 = 2u * p;
            const size_t r1 = 2u * p + 1u;
            strata::quant::sk::dequant_sf4_row(
                w2_p + r0 * (inter / 2u), w2_s + r0 * (inter / 16u), w2_outer, pair_f32.data(), inter);
            strata::quant::sk::dequant_sf4_row(
                w2_p + r1 * (inter / 2u), w2_s + r1 * (inter / 16u), w2_outer, pair_f32.data() + inter, inter);
            uint8_t* dp = dst_dn + p * pair_b_s3_dn;
            for (size_t s = 0; s < n_sb_dn; ++s) {
                encode_s3k_superblock_mse(pair_f32.data() + s * 256u, dp + s * sb_s3);
            }
        }
    } else {
        // S4_K -> S3_K: preserve exact MACaQ smoothquant vector sq_xd_dn[inter] verbatim!
        std::memcpy(out_body.data(), in_body, 2u * inter);
        const size_t sb_s4 = strata::quant::sk::superblock_bytes(SkCode::S4_K);  // 134
        const size_t row_b_s4_gu = n_sb_gu * sb_s4;
        const size_t pair_b_s4_dn = n_sb_dn * sb_s4;

        const uint8_t* src_gu = in_body + 2u * inter;
        for (size_t r = 0; r < 2u * inter; ++r) {
            strata::quant::sk::dequant_sk_row(src_gu + r * row_b_s4_gu, SkCode::S4_K, row_f32.data(), hidden);
            uint8_t* dr = dst_gu + r * row_b_s3_gu;
            for (size_t s = 0; s < n_sb_gu; ++s) {
                encode_s3k_superblock_mse(row_f32.data() + s * 256u, dr + s * sb_s3);
            }
        }

        const uint8_t* src_dn = in_body + in_ent.dn_offset;
        for (size_t p = 0; p < hidden / 2u; ++p) {
            strata::quant::sk::dequant_sk_row(src_dn + p * pair_b_s4_dn, SkCode::S4_K, pair_f32.data(), 2u * inter);
            uint8_t* dp = dst_dn + p * pair_b_s3_dn;
            for (size_t s = 0; s < n_sb_dn; ++s) {
                encode_s3k_superblock_mse(pair_f32.data() + s * 256u, dp + s * sb_s3);
            }
        }
    }

    out_ent.gu_code_id = static_cast<uint8_t>(SkCode::S3_K);
    out_ent.dn_code_id = static_cast<uint8_t>(SkCode::S3_K);
    out_ent.dn_offset  = new_dn_off;
    out_ent.body_bytes = new_body_bytes;
}

}  // namespace

int main(int argc, char** argv) {
    std::string in_path, out_path;
    bool keep_coda = false;
    int n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) n_threads = 8;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--in" || a == "-i") && i + 1 < argc) in_path = argv[++i];
        else if ((a == "--out" || a == "-o") && i + 1 < argc) out_path = argv[++i];
        else if (a == "--keep-coda") keep_coda = true;
        else if ((a == "--threads" || a == "-t") && i + 1 < argc) n_threads = std::max(1, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "Usage: %s --in <SQ-15.strata> --out <SQ-12.strata> [--keep-coda] [--threads N]\n", argv[0]);
            return 1;
        }
    }
    if (in_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "Usage: %s --in <SQ-15.strata> --out <SQ-12.strata> [--keep-coda] [--threads N]\n", argv[0]);
        return 1;
    }

    if (!selfcheck_s3k_codec()) return 2;

    StrataFileReader rd;
    std::string err;
    if (!rd.open(in_path, err)) {
        std::fprintf(stderr, "[repack] open failed: %s\n", err.c_str());
        return 1;
    }
    ink2::InkManifest man;
    if (!ink2::parse_manifest(rd.manifest_json(), man, err, false)) {
        std::fprintf(stderr, "[repack] manifest parse failed: %s\n", err.c_str());
        return 1;
    }

    const size_t hidden = static_cast<size_t>(man.c.hidden);
    const size_t inter  = static_cast<size_t>(man.c.inter);
    const size_t n_exp_per_layer = static_cast<size_t>(man.c.n_experts);
    const auto& in_dir = rd.expert_directory();
    const size_t n_total = in_dir.size();

    std::printf("[repack] Input: %s (%zu experts, hidden=%zu, inter=%zu, keep_coda=%d, threads=%d)\n",
                in_path.c_str(), n_total, hidden, inter, (int)keep_coda, n_threads);

    // Read original 4 KiB header frame so magic/flags/profile/engram_sha256/section flags stay identical
    uint8_t header_frame[STRATA_PAGE_SIZE]{};
    if (!rd.read_at(0, header_frame, STRATA_PAGE_SIZE, err)) {
        std::fprintf(stderr, "[repack] read header failed: %s\n", err.c_str());
        return 1;
    }

    std::FILE* out_fp = std::fopen(out_path.c_str(), "wb");
    if (!out_fp) {
        std::fprintf(stderr, "[repack] failed to open output: %s\n", out_path.c_str());
        return 1;
    }

    // Reserve page 0 for header
    if (!write_all(out_fp, header_frame, STRATA_PAGE_SIZE)) {
        std::fprintf(stderr, "[repack] write header placeholder failed\n");
        std::fclose(out_fp);
        return 1;
    }
    uint64_t cur_off = STRATA_PAGE_SIZE;

    StrataSectionEntry out_secs[STRATA_NUM_SECTIONS]{};
    for (int i = 0; i < STRATA_NUM_SECTIONS; ++i) {
        out_secs[i] = rd.header().sections[i];
    }

    auto copy_section_verbatim = [&](int sid) -> bool {
        const auto& src = rd.header().sections[sid];
        if (src.byte_length == 0) {
            out_secs[sid].file_offset = 0;
            out_secs[sid].byte_length = 0;
            return true;
        }
        if (!pad_to_page(out_fp, cur_off)) return false;
        out_secs[sid].file_offset = cur_off;
        out_secs[sid].byte_length = src.byte_length;
        ink2::Sha256 h;
        std::vector<uint8_t> buf(16 * 1024 * 1024);
        uint64_t rem = src.byte_length;
        uint64_t roff = src.file_offset;
        while (rem > 0) {
            size_t n = static_cast<size_t>(std::min<uint64_t>(rem, buf.size()));
            if (!rd.read_at(roff, buf.data(), n, err)) return false;
            if (!write_all(out_fp, buf.data(), n)) return false;
            h.update(buf.data(), n);
            roff += n;
            rem -= n;
            cur_off += n;
        }
        auto dig = h.finish();
        std::memcpy(out_secs[sid].sha256, dig.data(), 32);
        return true;
    };

    const auto t0 = std::chrono::steady_clock::now();

    // 1. Copy Section 1 (trunk + DSpark) and Section 2 verbatim
    if (!copy_section_verbatim(1) || !copy_section_verbatim(2)) {
        std::fprintf(stderr, "[repack] failed copying trunk/section2: %s\n", err.c_str());
        std::fclose(out_fp);
        return 1;
    }
    std::printf("[repack] Copied Section 1 trunk (%.2f GiB)\n",
                static_cast<double>(out_secs[1].byte_length) / 1073741824.0);

    // 2. Repack Section 4 (expert bodies) in parallel batches of 64 experts
    if (!pad_to_page(out_fp, cur_off)) { std::fclose(out_fp); return 1; }
    const uint64_t sec4_start = cur_off;
    out_secs[4].file_offset = sec4_start;

    std::vector<StrataExpertEntry> out_dir(n_total);
    ink2::Sha256 sec4_sha;
    uint64_t sec4_pos = 0;
    size_t n_conv_sf4 = 0, n_conv_s4k = 0, n_kept = 0;

    constexpr size_t kBatch = 64;
    static const uint8_t pad64[64]{};

    for (size_t b0 = 0; b0 < n_total; b0 += kBatch) {
        const size_t b_cnt = std::min(kBatch, n_total - b0);
        std::vector<std::vector<uint8_t>> in_bodies(b_cnt);
        std::vector<std::vector<uint8_t>> out_bodies(b_cnt);
        for (size_t i = 0; i < b_cnt; ++i) {
            if (!rd.read_expert_body(b0 + i, in_bodies[i], err)) {
                std::fprintf(stderr, "[repack] read expert %zu failed: %s\n", b0 + i, err.c_str());
                std::fclose(out_fp);
                return 1;
            }
        }

        std::atomic<size_t> next_i{0};
        const int workers = std::min<int>(n_threads, static_cast<int>(b_cnt));
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (int w = 0; w < workers; ++w) {
            pool.emplace_back([&]() {
                while (true) {
                    size_t i = next_i.fetch_add(1, std::memory_order_relaxed);
                    if (i >= b_cnt) break;
                    const size_t idx = b0 + i;
                    const int lay = static_cast<int>(idx / n_exp_per_layer);
                    repack_one_expert(
                        in_bodies[i].data(), in_dir[idx], hidden, inter, keep_coda, lay,
                        out_bodies[i], out_dir[idx]);
                }
            });
        }
        for (auto& th : pool) th.join();

        for (size_t i = 0; i < b_cnt; ++i) {
            const size_t idx = b0 + i;
            if (in_dir[idx].gu_code_id == static_cast<uint8_t>(SkCode::SF4)) ++n_conv_sf4;
            else if (!keep_coda && in_dir[idx].gu_code_id == static_cast<uint8_t>(SkCode::S4_K)
                     && in_dir[idx].dn_code_id == static_cast<uint8_t>(SkCode::S4_K)) ++n_conv_s4k;
            else ++n_kept;

            const uint64_t aligned = align_up(sec4_pos, 64u);
            if (aligned > sec4_pos) {
                const size_t p = static_cast<size_t>(aligned - sec4_pos);
                if (!write_all(out_fp, pad64, p)) { std::fclose(out_fp); return 1; }
                sec4_sha.update(pad64, p);
                sec4_pos += p;
                cur_off += p;
            }
            out_dir[idx].body_offset = sec4_pos;
            const auto& body = out_bodies[i];
            if (!write_all(out_fp, body.data(), body.size())) { std::fclose(out_fp); return 1; }
            sec4_sha.update(body.data(), body.size());
            sec4_pos += body.size();
            cur_off += body.size();
        }
        if ((b0 + b_cnt) % 512 == 0 || b0 + b_cnt == n_total) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[repack] Experts %zu / %zu (%.1fs, sec4=%.2f GiB)\n",
                        b0 + b_cnt, n_total, sec, static_cast<double>(sec4_pos) / 1073741824.0);
            std::fflush(stdout);
        }
    }
    out_secs[4].byte_length = sec4_pos;
    {
        auto dig = sec4_sha.finish();
        std::memcpy(out_secs[4].sha256, dig.data(), 32);
    }

    // 3. Copy Section 0 (ZSTD manifest) verbatim
    if (!copy_section_verbatim(0)) {
        std::fprintf(stderr, "[repack] failed copying section 0\n");
        std::fclose(out_fp);
        return 1;
    }

    // 4. Write updated Section 3 (StrataExpertEntry[N] + optional vram_load_order[N])
    if (!pad_to_page(out_fp, cur_off)) { std::fclose(out_fp); return 1; }
    out_secs[3].file_offset = cur_off;
    const bool has_vram_order = ((out_secs[3].flags & 1u) != 0) && !rd.vram_load_order().empty();
    const size_t dir_bytes = out_dir.size() * sizeof(StrataExpertEntry);
    const size_t ord_bytes = has_vram_order ? (rd.vram_load_order().size() * sizeof(uint16_t)) : 0;
    ink2::Sha256 sec3_sha;
    if (!write_all(out_fp, out_dir.data(), dir_bytes)) { std::fclose(out_fp); return 1; }
    sec3_sha.update(reinterpret_cast<const uint8_t*>(out_dir.data()), dir_bytes);
    cur_off += dir_bytes;
    if (has_vram_order) {
        if (!write_all(out_fp, rd.vram_load_order().data(), ord_bytes)) { std::fclose(out_fp); return 1; }
        sec3_sha.update(reinterpret_cast<const uint8_t*>(rd.vram_load_order().data()), ord_bytes);
        cur_off += ord_bytes;
    }
    out_secs[3].byte_length = dir_bytes + ord_bytes;
    {
        auto dig = sec3_sha.finish();
        std::memcpy(out_secs[3].sha256, dig.data(), 32);
    }

    // 5. Copy Section 5 (if any) verbatim
    if (!copy_section_verbatim(5)) {
        std::fprintf(stderr, "[repack] failed copying section 5\n");
        std::fclose(out_fp);
        return 1;
    }

    // 6. Patch header frame with updated section entries & write page 0
    for (int i = 0; i < STRATA_NUM_SECTIONS; ++i) {
        std::memcpy(header_frame + 0x10 + i * sizeof(StrataSectionEntry), &out_secs[i], sizeof(StrataSectionEntry));
    }
    if (!seek_abs(out_fp, 0) || !write_all(out_fp, header_frame, STRATA_PAGE_SIZE)) {
        std::fprintf(stderr, "[repack] failed writing final header\n");
        std::fclose(out_fp);
        return 1;
    }
    std::fclose(out_fp);

    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[repack] DONE in %.1fs -> %s (%.2f GB / %.2f GiB | converted: %zu SF4->S3_K, %zu S4_K->S3_K, kept: %zu)\n",
                total_s, out_path.c_str(),
                static_cast<double>(cur_off) / 1e9,
                static_cast<double>(cur_off) / 1073741824.0,
                n_conv_sf4, n_conv_s4k, n_kept);
    return 0;
}

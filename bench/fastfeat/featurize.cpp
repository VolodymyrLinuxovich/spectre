// Spectrum binning for Spectre, in C++.
//
// Same result as the Python loop in SparseSpectra.__init__ (bench/baseline/model.py):
//   keep peaks with 0 <= mz < max_mz
//   bin  = int64(mz / bin_width)                (truncation, mz >= 0 so this is floor)
//   v    = sqrt(max(intensity, 0) / max(max intensity of kept peaks, 1e-12))
//   per spectrum and bin, keep the largest v, bins ascending, empty spectra skipped
//
// Plain C ABI so Python can load it with ctypes. No pybind11, no build system.
// Spectra are split across threads; each thread writes into its own slice of the output,
// found by a first counting pass, so the output order matches the Python loop exactly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace {

// Bins and values of one spectrum, sorted by bin, max aggregated. Returns the unique count.
int64_t bin_one(const double* mz, const double* it, int64_t n, double bin_width, double max_mz,
                std::vector<std::pair<int64_t, double>>& buf) {
    buf.clear();
    double top = -INFINITY;
    for (int64_t k = 0; k < n; ++k) {
        if (mz[k] >= 0.0 && mz[k] < max_mz) top = std::max(top, it[k]);
    }
    if (top == -INFINITY) return 0;  // no kept peaks
    const double denom = std::max(top, 1e-12);
    for (int64_t k = 0; k < n; ++k) {
        if (!(mz[k] >= 0.0 && mz[k] < max_mz)) continue;
        const int64_t b = static_cast<int64_t>(mz[k] / bin_width);
        const double v = std::sqrt(std::max(it[k], 0.0) / denom);
        buf.emplace_back(b, v);
    }
    std::sort(buf.begin(), buf.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    int64_t w = 0;
    for (size_t k = 0; k < buf.size(); ++k) {
        if (w > 0 && buf[w - 1].first == buf[k].first) {
            buf[w - 1].second = std::max(buf[w - 1].second, buf[k].second);
        } else {
            // agg starts at 0 in the Python version, so a bin's value is never below 0
            buf[w++] = {buf[k].first, std::max(0.0, buf[k].second)};
        }
    }
    return w;
}

}  // namespace

extern "C" {

// offsets: n_spectra + 1 entries into mz / intensity (CSR layout).
// Outputs must hold at least offsets[n_spectra] entries (an upper bound on unique bins).
// Returns the number of (row, bin, value) triples written.
int64_t spectre_featurize(const int64_t* offsets, const double* mz, const double* intensity,
                          int64_t n_spectra, double bin_width, double max_mz,
                          int64_t* out_row, int64_t* out_bin, double* out_val, int n_threads) {
    if (n_spectra <= 0) return 0;
    if (n_threads < 1) n_threads = 1;
    n_threads = static_cast<int>(std::min<int64_t>(n_threads, n_spectra));

    // pass 1: unique bins per spectrum
    std::vector<int64_t> count(n_spectra, 0);
    auto count_range = [&](int64_t lo, int64_t hi) {
        std::vector<std::pair<int64_t, double>> buf;
        for (int64_t i = lo; i < hi; ++i) {
            const int64_t a = offsets[i], n = offsets[i + 1] - a;
            count[i] = bin_one(mz + a, intensity + a, n, bin_width, max_mz, buf);
        }
    };
    // pass 2: write each spectrum at its prefix-sum position
    std::vector<int64_t> start(n_spectra + 1, 0);
    auto write_range = [&](int64_t lo, int64_t hi) {
        std::vector<std::pair<int64_t, double>> buf;
        for (int64_t i = lo; i < hi; ++i) {
            const int64_t a = offsets[i], n = offsets[i + 1] - a;
            const int64_t m = bin_one(mz + a, intensity + a, n, bin_width, max_mz, buf);
            int64_t p = start[i];
            for (int64_t k = 0; k < m; ++k, ++p) {
                out_row[p] = i;
                out_bin[p] = buf[k].first;
                out_val[p] = buf[k].second;
            }
        }
    };
    auto parallel = [&](auto&& fn) {
        std::vector<std::thread> ts;
        const int64_t step = (n_spectra + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; ++t) {
            const int64_t lo = t * step, hi = std::min(n_spectra, lo + step);
            if (lo < hi) ts.emplace_back(fn, lo, hi);
        }
        for (auto& th : ts) th.join();
    };

    parallel(count_range);
    for (int64_t i = 0; i < n_spectra; ++i) start[i + 1] = start[i] + count[i];
    parallel(write_range);
    return start[n_spectra];
}

}  // extern "C"

// verify.cpp: recompute a Referee verdict from raw ranks, without trusting any Python.
//
// The Referee (bench/referee.py) decides whether a claimed model improvement is real.
// It writes down the verdict and the numbers behind it. This program takes the raw
// per-spectrum ranks those numbers were computed from (exported by verify/export.py),
// redoes every statistic from scratch, and applies the same decision rules in the
// same order:
//
//   NOISE    the val gain must beat 2x the incumbent's seed std, have a bootstrap 95%
//            CI above zero, and be positive on every seed.
//   PRIOR    with zeroed spectra, the gain must shrink below half.
//   LEAK     with shuffled train labels, the gain must shrink below half.
//   OVERFIT  on the hidden test fold, the bootstrap CI must stay above zero.
//   PROMOTED otherwise.
//
// Naive verdicts (the unguarded baseline loop) are a single rule: ACCEPTED if the
// seed 0 val gain is positive, else REJECTED.
//
// It prints claimed vs recomputed for each number and exits 0 only if the verdict
// matches and every number agrees within tolerance.
//
// Tolerances. Everything except the CI bounds is deterministic arithmetic, so it must
// match to 1e-9. The CI comes from random resampling, and this program uses its own
// RNG (splitmix64), not numpy's. The bounds therefore differ by Monte Carlo error,
// roughly 0.06 bootstrap standard deviations per side at B = 2000. We accept a CI bound
// if it is within 15% of the interval width (about 0.6 bootstrap SDs). If a recomputed
// lower bound sits that close to zero, the sign test is too close to call with a
// different RNG, so we take the claimed bound's sign and say so in the output.
//
// Usage:  verify <case.txt>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kTopK = 20;
constexpr int kBootstrapResamples = 2000;
constexpr double kExactTolerance = 1e-9;
constexpr double kCiRelativeTolerance = 0.15;

// ------------------------------------------------------------------ case file

struct RunKey {
    std::string who;    // "cand" or "inc"
    std::string split;  // "val" or "test"
    int seed = 0;
    std::string mode;   // "plain", "shuffle" or "ablate"

    bool operator<(const RunKey& other) const {
        return std::tie(who, split, seed, mode) <
               std::tie(other.who, other.split, other.seed, other.mode);
    }
};

struct Case {
    std::string name;
    std::map<std::string, std::string> claims;
    std::map<RunKey, std::vector<int>> runs;

    std::optional<double> claimed_number(const std::string& field) const {
        auto it = claims.find(field);
        if (it == claims.end()) return std::nullopt;
        return std::stod(it->second);
    }

    const std::vector<int>* find_run(const std::string& who, const std::string& split,
                                     int seed, const std::string& mode) const {
        auto it = runs.find(RunKey{who, split, seed, mode});
        return it == runs.end() ? nullptr : &it->second;
    }
};

bool read_case(std::istream& in, Case& c) {
    std::string tag;
    while (in >> tag) {
        if (tag == "CASE") {
            in >> c.name;
        } else if (tag == "CLAIM") {
            std::string field, value;
            in >> field >> value;
            c.claims[field] = value;
        } else if (tag == "RUN") {
            RunKey key;
            std::size_t n = 0;
            in >> key.who >> key.split >> key.seed >> key.mode >> n;
            std::vector<int> ranks(n);
            for (int& r : ranks) in >> r;
            if (!in) return false;
            c.runs[key] = ranks;
        } else if (tag == "END") {
            return true;
        } else {
            std::cerr << "unknown record: " << tag << "\n";
            return false;
        }
    }
    return false;
}

// ------------------------------------------------------------------ metrics

// Reciprocal rank of one spectrum: 1/(r+1) if the true molecule is in the top 20.
double reciprocal_rank(int rank) {
    return (rank >= 0 && rank < kTopK) ? 1.0 / (rank + 1) : 0.0;
}

std::vector<double> reciprocal_ranks(const std::vector<int>& ranks) {
    std::vector<double> rr;
    rr.reserve(ranks.size());
    for (int r : ranks) rr.push_back(reciprocal_rank(r));
    return rr;
}

double mean(const std::vector<double>& xs) {
    double total = 0.0;
    for (double x : xs) total += x;
    return xs.empty() ? 0.0 : total / static_cast<double>(xs.size());
}

double mrr_at_20(const std::vector<int>& ranks) { return mean(reciprocal_ranks(ranks)); }

// Sample standard deviation (divide by n-1). With 3 seeds the population std would
// understate retraining noise by about 18%, which would make NOISE too lenient.
double sample_std(const std::vector<double>& xs) {
    double m = mean(xs);
    double squares = 0.0;
    for (double x : xs) squares += (x - m) * (x - m);
    return std::sqrt(squares / static_cast<double>(xs.size() - 1));
}

// ------------------------------------------------------------------ bootstrap

// splitmix64: a small, well known 64-bit generator. Good enough for resampling and
// easy to reimplement in any language, which is the point of this file.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Uniform integer in [0, n). Rejection sampling avoids modulo bias.
    std::size_t below(std::size_t n) {
        const std::uint64_t bound = static_cast<std::uint64_t>(n);
        const std::uint64_t limit = UINT64_MAX - UINT64_MAX % bound;
        std::uint64_t x = next();
        while (x >= limit) x = next();
        return static_cast<std::size_t>(x % bound);
    }

private:
    std::uint64_t state_;
};

// Percentile with linear interpolation between order statistics (numpy's default).
double percentile(std::vector<double> xs, double pct) {
    std::sort(xs.begin(), xs.end());
    double position = pct / 100.0 * static_cast<double>(xs.size() - 1);
    std::size_t below = static_cast<std::size_t>(std::floor(position));
    std::size_t above = std::min(below + 1, xs.size() - 1);
    double fraction = position - static_cast<double>(below);
    return xs[below] + fraction * (xs[above] - xs[below]);
}

struct Interval {
    double lo = 0.0;
    double hi = 0.0;
};

// Paired bootstrap over spectra. Each d_i is already the candidate minus incumbent
// difference on the same spectrum, so resampling the d_i keeps the pairing: a hard
// spectrum is hard for both models and that shared difficulty cancels. Resampling the
// two models' scores independently would ignore that and give a much wider interval.
Interval bootstrap_ci(const std::vector<double>& diffs) {
    SplitMix64 rng(0);
    std::vector<double> resample_means;
    resample_means.reserve(kBootstrapResamples);
    for (int b = 0; b < kBootstrapResamples; ++b) {
        double total = 0.0;
        for (std::size_t i = 0; i < diffs.size(); ++i) total += diffs[rng.below(diffs.size())];
        resample_means.push_back(total / static_cast<double>(diffs.size()));
    }
    return {percentile(resample_means, 2.5), percentile(resample_means, 97.5)};
}

// ------------------------------------------------------------------ paired gain

struct PairedGain {
    double gain = 0.0;                // mean over spectra of d_i
    std::vector<double> per_seed;     // MRR(cand, s) - MRR(inc, s) for each seed
    std::vector<double> inc_mrr;      // incumbent MRR per seed, for the noise band
    Interval ci;
};

std::vector<int> seeds_with_both(const Case& c, const std::string& split) {
    std::vector<int> seeds;
    for (const auto& entry : c.runs) {
        const RunKey& k = entry.first;
        if (k.who == "cand" && k.split == split && k.mode == "plain" &&
            c.find_run("inc", split, k.seed, "plain") != nullptr) {
            seeds.push_back(k.seed);
        }
    }
    return seeds;
}

// d_i averages the candidate-minus-incumbent difference over seeds before
// bootstrapping. Averaging first removes most retraining noise from each d_i, so the
// bootstrap then measures only spectrum sampling noise. Seed noise is tested
// separately by the 2x seed std band.
PairedGain paired_gain(const Case& c, const std::string& split, const std::vector<int>& seeds) {
    PairedGain result;
    std::vector<double> diffs;
    for (int seed : seeds) {
        std::vector<double> cand = reciprocal_ranks(*c.find_run("cand", split, seed, "plain"));
        std::vector<double> inc = reciprocal_ranks(*c.find_run("inc", split, seed, "plain"));
        if (diffs.empty()) diffs.assign(cand.size(), 0.0);
        for (std::size_t i = 0; i < cand.size(); ++i) diffs[i] += (cand[i] - inc[i]) / seeds.size();
        result.per_seed.push_back(mean(cand) - mean(inc));
        result.inc_mrr.push_back(mean(inc));
    }
    result.gain = mean(diffs);
    result.ci = bootstrap_ci(diffs);
    return result;
}

std::optional<double> canary_gain(const Case& c, const std::string& mode) {
    const std::vector<int>* cand = c.find_run("cand", "val", 0, mode);
    const std::vector<int>* inc = c.find_run("inc", "val", 0, mode);
    if (!cand || !inc) return std::nullopt;
    return mrr_at_20(*cand) - mrr_at_20(*inc);
}

// ------------------------------------------------------------------ report

class Report {
public:
    explicit Report(const Case& c) : case_(c) {}

    void exact(const std::string& field, double recomputed) {
        add(field, recomputed, kExactTolerance);
    }

    void interval(const std::string& prefix, Interval recomputed) {
        double width = recomputed.hi - recomputed.lo;
        double tolerance = std::max(1e-6, kCiRelativeTolerance * width);
        add(prefix + "_lo", recomputed.lo, tolerance, "bootstrap");
        add(prefix + "_hi", recomputed.hi, tolerance, "bootstrap");
    }

    void note(const std::string& text) { notes_.push_back(text); }

    bool numbers_agree() const { return all_agree_; }

    void print(const std::string& claimed_verdict, const std::string& recomputed_verdict) const {
        std::printf("case %s\n", case_.name.c_str());
        std::printf("  %-18s %14s %14s %12s  %s\n", "field", "claimed", "recomputed", "tolerance", "status");
        for (const std::string& row : rows_) std::printf("%s\n", row.c_str());
        std::printf("  %-18s %14s %14s %12s  %s\n", "verdict", claimed_verdict.c_str(),
                    recomputed_verdict.c_str(), "exact",
                    claimed_verdict == recomputed_verdict ? "ok" : "DIFF");
        for (const std::string& n : notes_) std::printf("  note: %s\n", n.c_str());
    }

private:
    void add(const std::string& field, double recomputed, double tolerance,
             const std::string& kind = "") {
        std::optional<double> claimed = case_.claimed_number(field);
        char claimed_text[32] = "-";
        if (claimed) std::snprintf(claimed_text, sizeof claimed_text, "%+.6f", *claimed);
        std::string status = "not claimed";
        if (claimed) {
            bool agree = std::fabs(*claimed - recomputed) <= tolerance;
            all_agree_ = all_agree_ && agree;
            status = agree ? "ok" : "DIFF";
        }
        char row[160];
        std::snprintf(row, sizeof row, "  %-18s %14s %+14.6f %12.1e  %s%s", field.c_str(),
                      claimed_text, recomputed, tolerance, status.c_str(),
                      kind.empty() ? "" : "  (own RNG)");
        rows_.push_back(row);
    }

    const Case& case_;
    std::vector<std::string> rows_;
    std::vector<std::string> notes_;
    bool all_agree_ = true;
};

// "Is the CI lower bound above zero?" with an honest answer when our RNG cannot tell.
bool lower_bound_positive(const Case& c, const std::string& field, Interval recomputed,
                          Report& report) {
    double tolerance = std::max(1e-6, kCiRelativeTolerance * (recomputed.hi - recomputed.lo));
    std::optional<double> claimed = c.claimed_number(field);
    if (std::fabs(recomputed.lo) > tolerance || !claimed) return recomputed.lo > 0;
    report.note(field + " is within bootstrap tolerance of 0; using the claimed sign");
    return *claimed > 0;
}

// ------------------------------------------------------------------ decision rules

std::string judge_naive(const Case& c, Report& report) {
    const std::vector<int>* cand = c.find_run("cand", "val", 0, "plain");
    const std::vector<int>* inc = c.find_run("inc", "val", 0, "plain");
    if (!cand || !inc) return "MISSING_RUNS";
    double gain = mrr_at_20(*cand) - mrr_at_20(*inc);
    report.exact("val_gain", gain);
    return gain > 0 ? "ACCEPTED" : "REJECTED";
}

std::string judge_guarded(const Case& c, Report& report) {
    // Stage NOISE: val, all seeds.
    std::vector<int> seeds = seeds_with_both(c, "val");
    if (seeds.size() < 2) return "MISSING_RUNS";
    PairedGain val = paired_gain(c, "val", seeds);
    double noise_band = 2.0 * sample_std(val.inc_mrr);
    report.exact("val_gain", val.gain);
    report.interval("val_ci", val.ci);
    report.exact("noise_band", noise_band);
    for (std::size_t i = 0; i < val.per_seed.size(); ++i)
        report.exact("per_seed_gain_" + std::to_string(i), val.per_seed[i]);

    bool beats_sampling_noise = lower_bound_positive(c, "val_ci_lo", val.ci, report);
    bool beats_seed_noise = val.gain > noise_band;
    bool every_seed_positive = *std::min_element(val.per_seed.begin(), val.per_seed.end()) > 0;
    if (!beats_sampling_noise || !beats_seed_noise || !every_seed_positive) return "NOISE";

    // Stages LEAK and PRIOR: canaries on seed 0. Each removes what a real gain must
    // come from (learnable labels, or the spectrum itself). A real gain should mostly
    // vanish. Half is the line: if at least half survives, the gain is mainly something
    // else, even allowing for noise in a single seed canary run.
    std::optional<double> shuffle = canary_gain(c, "shuffle");
    std::optional<double> ablate = canary_gain(c, "ablate");
    if (!shuffle || !ablate) return "MISSING_RUNS";
    report.exact("shuffle_gain", *shuffle);
    report.exact("ablate_gain", *ablate);
    // PRIOR first: a gain that survives blank spectra is non-spectral, whether or not it is learned.
    if (*ablate >= 0.5 * val.gain) return "PRIOR";
    if (*shuffle >= 0.5 * val.gain) return "LEAK";

    // Stage OVERFIT: the same paired bootstrap on the hidden test fold.
    std::vector<int> test_seeds = seeds_with_both(c, "test");
    if (test_seeds.empty()) return "MISSING_RUNS";
    PairedGain test = paired_gain(c, "test", test_seeds);
    report.exact("test_gain", test.gain);
    report.interval("test_ci", test.ci);
    return lower_bound_positive(c, "test_ci_lo", test.ci, report) ? "PROMOTED" : "OVERFIT";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: verify <case.txt>\n";
        return 2;
    }
    std::ifstream file(argv[1]);
    Case c;
    if (!file || !read_case(file, c)) {
        std::cerr << "could not read case file " << argv[1] << "\n";
        return 2;
    }

    auto claimed = c.claims.find("verdict");
    std::string claimed_verdict = claimed == c.claims.end() ? "?" : claimed->second;
    bool naive = claimed_verdict == "ACCEPTED" || claimed_verdict == "REJECTED";

    Report report(c);
    std::string recomputed = naive ? judge_naive(c, report) : judge_guarded(c, report);
    report.print(claimed_verdict, recomputed);

    bool match = recomputed == claimed_verdict && report.numbers_agree();
    std::printf("%s\n\n", match ? "VERDICT MATCH" : "VERDICT MISMATCH");
    return match ? 0 : 1;
}

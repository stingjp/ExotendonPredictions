import numpy as np
from scipy.stats import shapiro, ttest_1samp, wilcoxon, rankdata

def _rank_biserial_from_diffs(diffs, zero_method="wilcox"):
    """
    Rank-biserial correlation (RBC) for Wilcoxon signed-rank test, computed from diffs.
    RBC = (W_plus - W_minus) / (W_plus + W_minus) = (2*W_plus / total_rank_sum) - 1

    Notes:
    - We rank |diffs| with average ranks for ties.
    - Handling zeros:
        * "wilcox": drop zeros (matches common Wilcoxon default behavior)
        * "pratt": include zeros in ranking but give them zero sign contribution (common interpretation)
        * "zsplit": split zero ranks half to + and half to - (rare; included)
    """
    diffs = np.asarray(diffs, dtype=float)

    is_zero = diffs == 0
    if zero_method == "wilcox":
        diffs_eff = diffs[~is_zero]
        if diffs_eff.size == 0:
            return np.nan
        abs_vals = np.abs(diffs_eff)
        ranks = rankdata(abs_vals, method="average")
        signs = np.sign(diffs_eff)
        w_plus = np.sum(ranks[signs > 0])
        w_minus = np.sum(ranks[signs < 0])
        denom = w_plus + w_minus
        return np.nan if denom == 0 else (w_plus - w_minus) / denom

    # For pratt and zsplit we rank all abs(diffs), including zeros
    abs_vals = np.abs(diffs)
    ranks = rankdata(abs_vals, method="average")

    signs = np.sign(diffs)
    w_plus = np.sum(ranks[signs > 0])
    w_minus = np.sum(ranks[signs < 0])

    if zero_method == "pratt":
        # zeros contribute nothing (neither + nor -)
        pass
    elif zero_method == "zsplit":
        # split the zero ranks evenly between + and -
        w_zero = np.sum(ranks[is_zero])
        w_plus += 0.5 * w_zero
        w_minus += 0.5 * w_zero
    else:
        raise ValueError("zero_method must be one of: 'wilcox', 'pratt', 'zsplit'.")

    denom = w_plus + w_minus
    return np.nan if denom == 0 else (w_plus - w_minus) / denom


def paired_test_auto(
    diffs,
    *,
    alpha_normality=0.05,
    alternative="two-sided",
    zero_method="wilcox",
    nan_policy="omit",
    n_comparisons=1,
):
    """
    Automatic paired test from a 1D array of paired differences (x - y):
      1) Shapiro–Wilk normality test on diffs
      2) If normal at alpha_normality -> paired t-test (one-sample t-test on diffs)
         else -> Wilcoxon signed-rank test
      3) Report effect size:
         - t-test: Cohen's d (paired) = mean(diffs) / sd(diffs, ddof=1)
         - Wilcoxon: rank-biserial correlation (RBC)

    Returns a results dict (easy to log or serialize).
    """
    diffs = np.asarray(diffs, dtype=float)

    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1.")

    if nan_policy == "omit":
        diffs = diffs[~np.isnan(diffs)]
    elif nan_policy != "raise" and np.isnan(diffs).any():
        raise ValueError("nan_policy must be 'omit' or 'raise'.")

    n = diffs.size
    if n < 3:
        raise ValueError("Need at least 3 paired differences (Shapiro–Wilk requires >=3).")

    # Shapiro–Wilk
    sh_w, sh_p = shapiro(diffs)
    is_normal = sh_p >= alpha_normality

    results = {
        "n": int(n),
        "diff_mean": float(np.mean(diffs)),
        "diff_sd": float(np.std(diffs, ddof=1)) if n > 1 else np.nan,
        "diff_median": float(np.median(diffs)),
        "diff_iqr": float(np.subtract(*np.percentile(diffs, [75, 25]))),
        "shapiro_W": float(sh_w),
        "shapiro_p": float(sh_p),
        "normal_at_alpha": bool(is_normal),
        "alpha_normality": float(alpha_normality),
        "chosen_test": None,
        "statistic": None,
        "p_value": None,
        "effect_size_name": None,
        "effect_size": None,
        "alternative": alternative,
        "zero_method": zero_method,
        "n_comparisons": int(n_comparisons),
        "p_value_bonferroni": None,
    }

    if is_normal:
        # Paired t-test == one-sample t-test on differences
        t_stat, p = ttest_1samp(diffs, popmean=0.0, alternative=alternative)

        sd = np.std(diffs, ddof=1)
        d = np.nan if sd == 0 else np.mean(diffs) / sd  # Cohen's d (paired)

        results.update({
            "chosen_test": "paired_t_test (ttest_1samp on diffs)",
            "statistic": float(t_stat),
            "p_value": float(p),
            "effect_size_name": "Cohen_d_paired",
            "effect_size": float(d) if np.isfinite(d) else np.nan,
        })
    else:
        w_stat, p = wilcoxon(
            diffs,
            alternative=alternative,
            zero_method=zero_method,
        )
        rbc = _rank_biserial_from_diffs(diffs, zero_method=zero_method)

        results.update({
            "chosen_test": "wilcoxon_signed_rank",
            "statistic": float(w_stat),
            "p_value": float(p),
            "effect_size_name": "rank_biserial_correlation",
            "effect_size": float(rbc) if np.isfinite(rbc) else np.nan,
        })

    results["p_value_bonferroni"] = float(min(results["p_value"] * n_comparisons, 1.0))

    return results


def format_report(res, *, digits=4):
    """Human-readable, single-block report."""
    n = res["n"]
    alt = res["alternative"]
    lines = []
    lines.append(f"n = {n}")
    lines.append(
        f"Differences: mean = {res['diff_mean']:.{digits}g}, "
        f"SD = {res['diff_sd']:.{digits}g}, "
        f"median = {res['diff_median']:.{digits}g}, "
        f"IQR = {res['diff_iqr']:.{digits}g}"
    )
    lines.append(
        f"Shapiro–Wilk normality: W = {res['shapiro_W']:.{digits}g}, "
        f"p = {res['shapiro_p']:.{digits}g} "
        f"(α = {res['alpha_normality']}) -> "
        f"{'normal' if res['normal_at_alpha'] else 'non-normal'}"
    )
    lines.append(f"Chosen test: {res['chosen_test']} (alternative = '{alt}')")
    lines.append(
        f"Test result: statistic = {res['statistic']:.{digits}g}, "
        f"p = {res['p_value']:.{digits}g}, "
        f"p_bonferroni = {res['p_value_bonferroni']:.{digits}g} "
        f"(n_comparisons = {res['n_comparisons']})"
    )
    lines.append(
        f"Effect size ({res['effect_size_name']}): {res['effect_size']:.{digits}g}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    # first metabolic stats
    # print('\nLong-compliant:'); diffs = np.array([-0.39, 0.55, -0.3, 0.03, -0.64, -0.62, -0.73, -0.06, 0.83, -1.77, -0.4])  # example (x - y)
    # print('\nShort-stiff:'); diffs = np.array([-1.64, -0.82, -1.17, -0.98, -1.29, -0.41, 0.73, 0.67, -0.88])  # example (x - y)
    # print('\nMedium-original:'); diffs = np.array([-1.94, -0.63, -0.76, -0.45, -1.08, -1.63, -0.68, -1.62, 1.3, -0.83, -1.14])  # example (x - y) 
    print('\nLong-stiff'); diffs = np.array([-1.01, -1.85, -0.72, 0.19, -0.81, 0.65, 0.02, -0.28, 1.5, -0.21])  # example (x - y)
    # print('\nBest-Individual:'); diffs = np.array([-1.94, -1.85, -1.17, -0.45, -1.08, -1.63, -0.73, -1.62, 0.67, -1.77, -1.14])  # example (x - y)

    # now 5k stats
    # print('\nRace-time:'); diffs = np.array([-11.00, 31.00, 3.00, -5.00, -20.00, -67.00, -16.00, -27.00, -13.00, -10.00])  # example (x - y)
    # print('\nHeart-rate:'); diffs = np.array([1.00, -11.00, -5.00, -1.00, -2.00, -1.00, -6.00, -3.00, -2.00, -9.00])  # example (x - y)
    # print('\nCadence:'); diffs = np.array([23.00, 24.00, 28.00, 11.00, 13.00, 18.00, 2.00, 11.00, 2.00, 27.00])  # example (x - y)
    
    #
    # print('\n5kspeed:'); diffs = np.array([0.042507937, -0.127082513, -0.011312644, 0.018724151, 0.077632217, 0.276864538, 0.06821137, 0.120966658, 0.062173351, 0.035667638])  # example (x - y)
    # 
    # print('\nRace-time w/o 2 outliers:'); diffs = np.array([-11.00, 3.00, -5.00, -20.00, -16.00, -27.00, -13.00, -10.00])  # example (x - y)
    


    res = paired_test_auto(
        diffs,
        alpha_normality=0.05,
        alternative="two-sided",
        zero_method="wilcox",   # or "pratt" if you want to keep zeros
        nan_policy="omit",
        n_comparisons=4,
    )

    print(format_report(res))


# --- IGNORE ---
'''
######################## 4 Metabolic comparisons ########################
Long-compliant:
n = 11
Differences: mean = -0.3182, SD = 0.6888, median = -0.39, IQR = 0.615
Shapiro–Wilk normality: W = 0.9465, p = 0.5995 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -1.532, p = 0.1565, p_bonferroni = 0.626 (n_comparisons = 4)
Effect size (Cohen_d_paired): -0.462

Short-stiff:
n = 9
Differences: mean = -0.6433, SD = 0.8333, median = -0.88, IQR = 0.76
Shapiro–Wilk normality: W = 0.8686, p = 0.1187 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -2.316, p = 0.04921, p_bonferroni = 0.1969 (n_comparisons = 4)
Effect size (Cohen_d_paired): -0.7721

Medium-original:
n = 11
Differences: mean = -0.86, SD = 0.8599, median = -0.83, IQR = 0.725
Shapiro–Wilk normality: W = 0.8573, p = 0.05311 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -3.317, p = 0.007785, p_bonferroni = 0.03114 (n_comparisons = 4)
Effect size (Cohen_d_paired): -1

Long-stiff
n = 10
Differences: mean = -0.252, SD = 0.9318, median = -0.245, IQR = 0.935
Shapiro–Wilk normality: W = 0.9863, p = 0.9899 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -0.8553, p = 0.4146, p_bonferroni = 1 (n_comparisons = 4)
Effect size (Cohen_d_paired): -0.2705


######################## 5 Metabolic comparisons ########################
Best-Individual:
n = 11
Differences: mean = -1.155, SD = 0.7703, median = -1.17, IQR = 0.795
Shapiro–Wilk normality: W = 0.8708, p = 0.07938 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -4.975, p = 0.0005576 (corrected=0.002788)
Effect size (Cohen_d_paired): -1.5

Long-stiff
n = 10
Differences: mean = -0.252, SD = 0.9318, median = -0.245, IQR = 0.935
Shapiro–Wilk normality: W = 0.9863, p = 0.9899 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -0.8553, p = 0.4146 (corrected=2.073)
Effect size (Cohen_d_paired): -0.2705

Medium-original:
n = 11
Differences: mean = -0.86, SD = 0.8599, median = -0.83, IQR = 0.725
Shapiro–Wilk normality: W = 0.8573, p = 0.05311 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -3.317, p = 0.007785 (corrected=0.03893)
Effect size (Cohen_d_paired): -1

Short-stiff:
n = 9
Differences: mean = -0.6433, SD = 0.8333, median = -0.88, IQR = 0.76
Shapiro–Wilk normality: W = 0.8686, p = 0.1187 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -2.316, p = 0.04921 (corrected=0.246)
Effect size (Cohen_d_paired): -0.7721

Long-compliant:
n = 11
Differences: mean = -0.3182, SD = 0.6888, median = -0.39, IQR = 0.615
Shapiro–Wilk normality: W = 0.9465, p = 0.5995 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -1.532, p = 0.1565 (corrected=0.7825)
Effect size (Cohen_d_paired): -0.462

######################## 3 comparisons 5k ########################
Race-time:
n = 10
Differences: mean = -13.5, SD = 24.58, median = -12, IQR = 12.75
Shapiro–Wilk normality: W = 0.9045, p = 0.2456 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -1.737, p = 0.1164 (corrected=0.3492)
Effect size (Cohen_d_paired): -0.5493

Heart-rate:
n = 10
Differences: mean = -3.9, SD = 3.814, median = -2.5, IQR = 4.5
Shapiro–Wilk normality: W = 0.9238, p = 0.3896 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -3.234, p = 0.01026 (corrected=0.03078)
Effect size (Cohen_d_paired): -1.023

Cadence:
n = 10
Differences: mean = 15.9, SD = 9.62, median = 15.5, IQR = 12.75
Shapiro–Wilk normality: W = 0.9147, p = 0.3147 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = 5.227, p = 0.0005444 (corrected=0.0016332)
Effect size (Cohen_d_paired): 1.653

5kspeed:
n = 10
Differences: mean = 0.05644, SD = 0.102, median = 0.05234, IQR = 0.05232
Shapiro–Wilk normality: W = 0.9143, p = 0.3118 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = 1.749, p = 0.1142
Effect size (Cohen_d_paired): 0.5531

Race-time w/o 2 outliers:
n = 8
Differences: mean = -12.38, SD = 9.133, median = -12, IQR = 8.25
Shapiro–Wilk normality: W = 0.9915, p = 0.997 (α = 0.05) -> normal
Chosen test: paired_t_test (ttest_1samp on diffs) (alternative = 'two-sided')
Test result: statistic = -3.832, p = 0.006437
Effect size (Cohen_d_paired): -1.355
'''
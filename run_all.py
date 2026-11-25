import pandas as pd
import matplotlib.pyplot as plt
from benchmarks.relational_and_lsm import run_relational_and_lsm_benchmarks
from benchmarks.vector_ann_benchmark import run_vector_benchmarks


def generate_charts(df_rel, df_vec):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(df_rel["structure"], df_rel["write_ops_sec"], color="#2563eb")
    ax1.set_title("Write Throughput (Records / sec)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Operations / sec (Higher is Better)")

    colors = ["#ef4444" if "Ball Tree" in alg else "#10b981" for alg in df_vec["algorithm"]]
    ax2.barh(df_vec["algorithm"], df_vec["p50_ms"], color=colors)
    ax2.set_xscale("log")
    ax2.axvline(1.0, color="#111827", linestyle="--", linewidth=1, label="1 ms Threshold")
    ax2.set_title("AI Vector Search p50 Latency in ms (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Latency ms (Lower is Better — Red: O(n) Degeneration)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("/app/results/benchmark_charts.png", dpi=200)
    print("Saved visualization to /app/results/benchmark_charts.png")


if __name__ == "__main__":
    rel_results = run_relational_and_lsm_benchmarks()
    df_rel = pd.DataFrame(rel_results)
    df_rel.to_csv("/app/results/relational_lsm_metrics.csv", index=False)
    print("\n=== RELATIONAL, TRIE & LSM-TREE RESULTS ===")
    print(df_rel.to_string(index=False))

    vec_results = run_vector_benchmarks()
    df_vec = pd.DataFrame(vec_results)
    df_vec.to_csv("/app/results/vector_ann_metrics.csv", index=False)
    print("\n=== HIGH-DIMENSIONAL AI VECTOR SEARCH RESULTS ===")
    print(df_vec.to_string(index=False))

    generate_charts(df_rel, df_vec)
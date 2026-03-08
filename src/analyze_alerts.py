"""
=============================================================================
Alert Log Analyzer & Visualizer
=============================================================================
Reads the alert log CSV and generates:
  - Timeline chart of alerts
  - Summary statistics
  - Hourly alert distribution
=============================================================================
Usage:
    python src/analyze_alerts.py
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from datetime import datetime


LOG_FILE = "outputs/alert_log.csv"
OUTPUT_DIR = "outputs/charts"


def load_log(log_file: str) -> pd.DataFrame:
    """Load and parse the alert log CSV."""
    if not os.path.exists(log_file):
        print(f"[ERROR] Log file not found: {log_file}")
        print("        Run safety_monitor.py first to generate alerts.")
        return None

    df = pd.read_csv(log_file)
    if df.empty:
        print("[INFO] No alerts logged yet.")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    return df


def plot_alert_timeline(df: pd.DataFrame, output_dir: str):
    """Plot alert events over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.scatter(df["timestamp"], df["persons_in_zone"],
               color="#e74c3c", s=80, zorder=5, label="Alert Event")
    ax.step(df["timestamp"], df["persons_in_zone"],
            color="#c0392b", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Persons in Danger Zone", fontsize=12)
    ax.set_title("⚠ Danger Zone Alert Timeline", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    path = os.path.join(output_dir, "alert_timeline.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.show()


def plot_hourly_distribution(df: pd.DataFrame, output_dir: str):
    """Plot hourly distribution of alerts."""
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(hourly.index, hourly.values, color="#3498db", edgecolor="white")

    # Highlight peak hour
    peak_hour = hourly.idxmax()
    bars[peak_hour].set_color("#e74c3c")

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Number of Alerts", fontsize=12)
    ax.set_title("Hourly Alert Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Annotation
    ax.annotate(f"Peak: {peak_hour:02d}:00\n({hourly[peak_hour]} alerts)",
                xy=(peak_hour, hourly[peak_hour]),
                xytext=(peak_hour + 1, hourly[peak_hour] + 0.5),
                fontsize=9, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    path = os.path.join(output_dir, "hourly_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.show()


def print_summary(df: pd.DataFrame):
    """Print a text summary of the alert log."""
    print("\n" + "="*50)
    print("  ALERT LOG SUMMARY")
    print("="*50)
    print(f"  Total alerts triggered : {len(df)}")

    if not df.empty:
        print(f"  Monitoring start       : {df['timestamp'].min()}")
        print(f"  Monitoring end         : {df['timestamp'].max()}")
        duration = df["timestamp"].max() - df["timestamp"].min()
        print(f"  Total duration         : {duration}")
        print(f"  Avg persons in zone    : {df['persons_in_zone'].mean():.2f}")
        print(f"  Max persons in zone    : {df['persons_in_zone'].max()}")
        print(f"  Peak alert hour        : {df['hour'].mode()[0]:02d}:00")

    print("="*50)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_log(LOG_FILE)
    if df is None or df.empty:
        print("[INFO] Generating sample data for demonstration...")
        df = generate_sample_data()

    print_summary(df)

    if len(df) > 0:
        plot_alert_timeline(df, OUTPUT_DIR)
        plot_hourly_distribution(df, OUTPUT_DIR)
        print(f"\n[DONE] Charts saved to: {OUTPUT_DIR}/")


def generate_sample_data() -> pd.DataFrame:
    """Generate sample alert data for demonstration."""
    np.random.seed(42)
    n = 25
    base = pd.Timestamp("2024-01-15 08:00:00")
    timestamps = [base + pd.Timedelta(minutes=int(x))
                  for x in np.cumsum(np.random.exponential(20, n))]
    persons = np.random.randint(1, 4, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "event": "DANGER_ZONE_ENTERED",
        "persons_in_zone": persons,
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    # Save sample
    os.makedirs("outputs", exist_ok=True)
    df[["timestamp", "event", "persons_in_zone"]].to_csv(LOG_FILE, index=False)
    print(f"[INFO] Sample data saved to: {LOG_FILE}")
    return df


if __name__ == "__main__":
    main()

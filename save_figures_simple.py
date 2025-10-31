# Save Individual Figures as HTML Files
# Run this after executing your notebook cells

import os

# Create figures directory
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

print("Saving individual figures as HTML files...")

# Save each figure individually (using actual figure names from notebook)
try:
    fig1.write_html(f"{figures_dir}/score_distribution.html")
    print("✅ Saved: score_distribution.html")
except Exception as e:
    print(f"❌ Error saving score distribution: {e}")

try:
    fig2.write_html(f"{figures_dir}/accuracy_vs_score.html")
    print("✅ Saved: accuracy_vs_score.html")
except Exception as e:
    print(f"❌ Error saving accuracy vs score: {e}")

# Note: The reviewer reliability figure is just called 'fig' (not fig3)
try:
    fig.write_html(f"{figures_dir}/reviewer_reliability.html")
    print("✅ Saved: reviewer_reliability.html")
except Exception as e:
    print(f"❌ Error saving reviewer reliability: {e}")

# Note: The reliability plot is also just called 'fig' (not fig4)
try:
    fig.write_html(f"{figures_dir}/reliability_plot.html")
    print("✅ Saved: reliability_plot.html")
except Exception as e:
    print(f"❌ Error saving reliability plot: {e}")

# Note: The utility vs threshold is fig1 (not fig5)
try:
    fig1.write_html(f"{figures_dir}/utility_vs_threshold.html")
    print("✅ Saved: utility_vs_threshold.html")
except Exception as e:
    print(f"❌ Error saving utility vs threshold: {e}")

# Note: The accuracy & review rate is fig2 (not fig6)
try:
    fig2.write_html(f"{figures_dir}/accuracy_review_rate.html")
    print("✅ Saved: accuracy_review_rate.html")
except Exception as e:
    print(f"❌ Error saving accuracy & review rate: {e}")

print(f"\n🎉 All figures saved to the '{figures_dir}' directory!")
print("You can now open these HTML files in any web browser to view the interactive plots.")

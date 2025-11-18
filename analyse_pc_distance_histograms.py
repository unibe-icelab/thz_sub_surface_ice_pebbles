from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from pyhelper import unibe

params = {"ytick.color":
              "black",

          "xtick.color":
              "black",

          "axes.labelcolor":
              "black",

          "axes.edgecolor":
              "black",

          "mathtext.fontset":
              "cm",

          "mathtext.rm":
              "Times New Roman",

          "font.family":
              "Helvetica Neue",

          "font.size":
              20}
plt.rcParams.update(params)


def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


if __name__ == "__main__":

    histogram_paths = {
        "Resolution Target": Path("output_pointclouds/1761566246.1480312_data_smoothed.csv"),
        "Dust Resolution Target": Path("output_pointclouds/1760700851.7425625_data_smoothed.csv"),
        "10 mm Spheres": Path("output_pointclouds/1760820898.0132043_data_smoothed.csv"),
        "Gaussian Surface 1.0 mm": Path("output_pointclouds/1761828786.7110758_data_smoothed_cropped.csv"),
        "Gaussian Surface 1.5 mm": Path("output_pointclouds/1761807038.002272_data_smoothed_cropped.csv"),
        "Gaussian Surface 2.0 mm": Path("output_pointclouds/1761589074.661844_data_smoothed_cropped.csv"),
        "Gaussian Surface 2.5 mm": Path("output_pointclouds/1761639251.877376_data_smoothed_cropped.csv"),
        "Gaussian Surface 5.0 mm": Path("output_pointclouds/1761723524.3136473_data_smoothed_cropped.csv"),

    }

    results = []

    for label, csv_path in histogram_paths.items():
        df = pd.read_csv(csv_path, delimiter=';')

        # Extract the bin ranges and values
        bin_edges = df[' Class start'].tolist() + [df[' Class end'].iloc[-1]]
        values = df[' Value'] / df[' Value'].max()  # Normalize the histogram

        # Calculate bin centers
        bin_centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))

        # Plot the bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, values, width=np.diff(bin_edges), edgecolor='black', color=unibe.BLUE, alpha=0.5)

        # Fit a Gaussian to the bar plot data
        popt, _ = curve_fit(gaussian, bin_centers, values, p0=[max(values), np.mean(values), np.std(values)])

        x_fit = np.linspace(-1, 1, 1000)
        y_fit = gaussian(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=unibe.RED, ls="--", linewidth=2, label='Gaussian Fit')

        # Plot FWHM lines
        # fwhm = 2.3548 * popt[2]
        # plt.axvline(popt[1] - fwhm / 2, color=unibe.BLUE, linestyle='--', label='FWHM')
        # plt.axvline(popt[1] + fwhm / 2, color=unibe.BLUE, linestyle='--')

        accuracy = popt[1]
        precision = popt[2]

        rmse = np.sqrt(accuracy ** 2 + precision ** 2)

        print(
            f"{label}      mu: {popt[1]:.2f} mm, Precision: {precision:.2f} mm, Accuracy: {accuracy:.2f} mm, RMSE: {rmse:.2f} mm")

        plt.xlim(-1, 1)

        # Add labels and title
        plt.xlabel(r'$\Delta$ Distance [mm]')
        plt.ylabel('Frequency, normalized')

        plt.legend()

        output_path = csv_path.with_stem(csv_path.stem + "_histogram_thesis").with_suffix(".png")
        plt.savefig(output_path)
        output_path = csv_path.with_stem(csv_path.stem + "_histogram_thesis").with_suffix(".pdf")
        plt.savefig(output_path)

        # Show the plot
        plt.show()

        precision = popt[2]

        results.append((label, popt[1], precision, rmse))

    # Write results to a LaTeX table
    latex_table_path = Path("results_table.tex")
    with open(latex_table_path, "w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|c|c|c}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Label} & \\textbf{Accuracy [mm]} & \\textbf{Precision [mm]} & \\textbf{RMSE [mm]} \\\\\n")
        f.write("\\midrule\n")
        for label, accuracy, precision, rmse in results:
            f.write(f"{label} & {accuracy:.2f} & {precision:.2f} & {rmse:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Gaussian fit results for point cloud histograms.}\n")
        f.write("\\label{tab:cloud_compare_hist_results}\n")
        f.write("\\end{table}\n")

    print(results)
    features_sizes = []
    accuracies = []
    for r in results:
        if "Gaussian" in r[0]:
            features_sizes.append(float(r[0].split(" ")[2]))
            accuracies.append(float(r[3]))

    plt.scatter(features_sizes, accuracies)
    plt.show()

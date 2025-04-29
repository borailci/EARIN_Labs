import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from contextlib import contextmanager
import io
from PIL import Image


class PlotManager:
    """
    A class to manage all visualization aspects of the wine classification project.
    Uses an object-oriented approach to handle plotting and saving figures.
    """

    def __init__(self, output_dir="", save_individual_plots=False):
        """Initialize PlotManager with optional output directory"""
        self.output_dir = output_dir
        # Set a consistent style for all plots
        sns.set_style("whitegrid")
        # Set a consistent color palette
        self.colors = sns.color_palette("viridis", 10)
        # Set default figure size
        plt.rcParams["figure.figsize"] = (10, 6)
        # Store all figures for multi-page display
        self.figures = []
        self.figure_titles = []
        # Whether to save individual plot files
        self.save_individual_plots = save_individual_plots

    def add_to_gallery(self, fig, title):
        """Add a figure to the collection for display in the multi-page viewer"""
        self.figures.append(fig)
        self.figure_titles.append(title)

    def save_all_figures(self, output_file="wine_analysis_report.pdf"):
        """Save all figures to a multi-page PDF"""
        # Since we're storing PIL Images, not matplotlib figures,
        # we need to create a PDF using PIL's functionality
        if not self.figures:
            print("No figures to save.")
            return

        # Create a list to hold the converted images
        images = []

        # Convert the first PIL image to RGB mode
        first_img = self.figures[0].convert("RGB")

        # Convert the remaining images (if any) to RGB mode and append to the list
        if len(self.figures) > 1:
            images = [img.convert("RGB") for img in self.figures[1:]]

        # Save the first image and append the remaining images to the PDF file
        first_img.save(
            f"{self.output_dir}{output_file}", save_all=True, append_images=images
        )
        print(f"Saved {len(self.figures)} figures to {output_file}")

    def show_gallery(self):
        """Display all figures in an interactive multi-page viewer"""
        if not self.figures:
            print("No figures to display.")
            return

        # Close all existing figures to ensure only the gallery is shown
        plt.close("all")

        # Make sure we're in interactive mode
        plt.ion()

        # Create figure navigator
        fig_idx = [0]  # Using list for mutable integer

        def key_event(e):
            if e.key == "right" or e.key == "n":
                fig_idx[0] = (fig_idx[0] + 1) % len(self.figures)
            elif e.key == "left" or e.key == "p":
                fig_idx[0] = (fig_idx[0] - 1) % len(self.figures)
            elif e.key == "q":
                plt.close("all")
                return

            update_display()

        def update_display():
            plt.figure(display_fig.number)
            plt.clf()
            # Convert PIL Image to numpy array if needed
            if hasattr(self.figures[fig_idx[0]], "convert"):
                # Ensure the PIL Image is properly processed for display
                img_array = np.array(self.figures[fig_idx[0]])
                plt.imshow(img_array)
            else:
                plt.imshow(self.figures[fig_idx[0]])
            plt.axis("off")
            plt.title(
                f"Figure {fig_idx[0]+1}/{len(self.figures)}: {self.figure_titles[fig_idx[0]]}",
                fontsize=16,
            )
            plt.draw()

        # Create a display figure
        display_fig = plt.figure(figsize=(12, 8))
        display_fig.canvas.mpl_connect("key_press_event", key_event)

        # Show first image using the same update function
        update_display()

        # Add instructions
        plt.figtext(
            0.5,
            0.01,
            "Navigation: Left/Right arrows or P/N keys to switch between plots, Q to quit",
            ha="center",
            fontsize=12,
        )

        print("\nMulti-page figure viewer opened.")
        print(
            "Navigation: Left/Right arrows or P/N keys to switch between plots, Q to quit"
        )

        # Use plt.show(block=True) to ensure the viewer stays open
        plt.show(block=True)

    @contextmanager
    def _create_figure(self, figsize=None, title=None):
        """Context manager for creating and handling figures"""
        if figsize:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

        try:
            yield fig
        finally:
            if title:
                # Only save individual file if requested
                if self.save_individual_plots:
                    plt.savefig(
                        f"{self.output_dir}{title.lower().replace(' ', '_')}.png"
                    )

                # Convert plot to image for gallery
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)

                # Add to gallery
                self.add_to_gallery(img, title)

                # Close the figure
                plt.close(fig)

    def plot_feature_distributions(self, dataframe):
        """Plot histograms of all features in the dataframe"""
        with self._create_figure(figsize=(15, 10), title="Feature Distributions"):
            dataframe.hist(bins=15, figsize=(15, 10))
            plt.suptitle("Feature Distributions", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title

    def plot_correlation_matrix(self, dataframe):
        """Plot correlation matrix heatmap"""
        correlation_matrix = dataframe.corr()

        with self._create_figure(figsize=(12, 10), title="Feature Correlation Matrix"):
            mask = np.zeros_like(correlation_matrix)
            mask[np.triu_indices_from(mask, k=1)] = True  # Only show lower triangle

            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="RdBu_r",
                fmt=".2f",
                linewidths=0.5,
                mask=mask,
                vmin=-1,
                vmax=1,
                cbar_kws={"label": "Correlation Coefficient"},
            )
            plt.title("Feature Correlation Matrix", fontsize=14)
            plt.tight_layout()

    def plot_feature_importance(
        self,
        feature_names,
        importance_values,
        title="Feature Importance (ANOVA F-test)",
    ):
        """Plot feature importance as horizontal bar chart"""
        # Create a DataFrame for the feature importance
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance_values}
        ).sort_values("Importance", ascending=False)

        with self._create_figure(figsize=(12, 8), title=title):
            ax = sns.barplot(
                x="Importance", y="Feature", data=importance_df, palette=self.colors
            )

            # Add value labels to the bars
            for i, value in enumerate(importance_df["Importance"]):
                ax.text(value + 0.1, i, f"{value:.2f}", va="center")

            plt.title(title, fontsize=14)
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")
            plt.tight_layout()

        return importance_df

    def plot_confusion_matrices(self, y_test, svm_preds, rf_preds):
        """Plot confusion matrices for SVM and Random Forest side by side"""
        # Calculate confusion matrices
        cm_svm = confusion_matrix(y_test, svm_preds)
        cm_rf = confusion_matrix(y_test, rf_preds)

        # Get class labels
        classes = sorted(np.unique(y_test))

        # Create figure with two subplots
        with self._create_figure(figsize=(14, 6), title="Confusion Matrices"):
            fig = plt.gcf()
            axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

            # Function to plot a single confusion matrix
            def plot_cm(cm, ax, title):
                # Normalize confusion matrix
                cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

                # Plot both raw counts and percentages
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

                # Add percentage labels
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j + 0.5,
                            i + 0.7,
                            f"{cm_norm[i, j]:.1%}",
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=9,
                        )

                ax.set_title(title, fontsize=14)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                ax.set_xticks(np.arange(len(classes)) + 0.5)
                ax.set_yticks(np.arange(len(classes)) + 0.5)
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)

            # Plot both confusion matrices
            plot_cm(cm_svm, axes[0], "SVM Confusion Matrix")
            plot_cm(cm_rf, axes[1], "Random Forest Confusion Matrix")

            plt.tight_layout()

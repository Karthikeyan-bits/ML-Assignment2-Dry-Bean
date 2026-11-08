# ============================================================
# ML ASSIGNMENT 2 - DRY BEAN CLASSIFICATION
# BLOCK 5.2 - MODEL LOADING AND CSV UPLOAD
# ============================================================

# Import Streamlit to create the interactive web application.
import streamlit as st

# Import Pandas to read and process uploaded CSV files.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Joblib to load previously trained machine learning models.
import joblib

# Import Path to locate files independently of the operating system.
from pathlib import Path


# ------------------------------------------------------------
# SECTION 1 - APPLICATION CONFIGURATION
# ------------------------------------------------------------

# Define the folder containing this application.
# This works in the BITS Virtual Lab and Streamlit Cloud.
BASE_DIR = Path(__file__).resolve().parent

# Define the folder containing the five saved model pipelines.
MODEL_DIR = BASE_DIR / "models"

# Define the location of the sample labeled test dataset.
TEST_DATA_PATH = BASE_DIR / "test_data.csv"

# Configure the browser tab and application layout.
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

# ------------------------------------------------------------
# APPLICATION THEME AND HERO BANNER
# ------------------------------------------------------------
# CSS affects presentation only; model computations are unchanged.
st.markdown("""
<style>
.stApp {background: #F4F7FA; color: #183342;}
[data-testid="stSidebar"] {background: #102D3D;}

[data-testid="stSidebar"] button * {color: inherit;}
.block-container {max-width: 1200px; padding-top: 1.4rem;}
h1,h2,h3 {color: #14394B; letter-spacing: -0.02em;}
[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DFE8EE;
  border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 8px rgba(16,45,61,.04);}
[data-testid="stMetricLabel"] {color: #526978;}
[data-testid="stMetricValue"] {color: #103F50;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
.hero {background: linear-gradient(115deg,#103447,#146D72); padding: 28px 32px;
 border-radius: 20px; color: white; margin-bottom: 22px;}
.hero-kicker {font-size: .76rem; font-weight: 750; letter-spacing: .13em; color:#A5F3D9;}
.hero-title {font-size: 2.1rem; font-weight: 800; margin: 7px 0 5px; color: white;}
.hero-sub {font-size: .94rem; color: #DCECF0;}
.hero-pills {margin-top: 17px; display:flex; gap:9px; flex-wrap:wrap;}
.hero-pill {border:1px solid #75BDBB; border-radius:20px; padding:5px 11px;
 font-size:.78rem; color:#E9FFFB;}

/* Sidebar: retain dark background and readable native control values. */
[data-testid="stSidebar"] {color: #FFFFFF;}
[data-testid="stSidebar"] button {
    background: #FFFFFF !important;
    color: #123247 !important;
    border: 1px solid #CBD5E1 !important;
}
[data-testid="stSidebar"] button * {color: #123247 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    border: 1px dashed #94A3B8 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
    color: #123247 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: #F1F5F9 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #102D3D !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] * {
    color: #102D3D !important;
    -webkit-text-fill-color: #102D3D !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    color: #102D3D !important;
    fill: #102D3D !important;
}

</style>
""", unsafe_allow_html=True)

# Main banner presents the assignment and dataset at a glance.
st.markdown("""
<div class="hero">
 <div class="hero-kicker">BITS PILANI · MACHINE LEARNING ASSIGNMENT 2</div>
 <div class="hero-title">Dry Bean Classification</div>
 <div class="hero-sub">Interactive evaluation of five pre-trained multiclass classifiers</div>
 <div class="hero-pills">
  <span class="hero-pill">5 Classifiers</span>
  <span class="hero-pill">16 Input Features</span>
  <span class="hero-pill">7 Bean Classes</span>
 </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# SECTION 2 - MODEL CONFIGURATION
# ------------------------------------------------------------

# Map the model names displayed in the application to their
# corresponding trained model filenames.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Define the name of the target column in the Dry Bean dataset.
TARGET_COLUMN = "Class"


# ------------------------------------------------------------
# SECTION 3 - LOAD SAVED CLASSIFICATION MODELS
# ------------------------------------------------------------

# Cache the loaded models so they are not repeatedly loaded
# from disk whenever the user interacts with the application.
@st.cache_resource
def load_models():

    # Create an empty dictionary to store the loaded models.
    loaded_models = {}

    # Load each previously trained model from the models folder.
    for model_name, filename in MODEL_FILES.items():

        # Construct the complete path to the saved model file.
        model_path = MODEL_DIR / filename

        # Check whether the required model file exists.
        if not model_path.is_file():

            # Stop loading and report the missing model file.
            raise FileNotFoundError(
                f"Missing model file: {model_path}"
            )

        # Load the previously fitted model pipeline.
        # Only load files created and trusted by this project.
        loaded_models[model_name] = joblib.load(model_path)

    # Return the dictionary containing all five loaded models.
    return loaded_models


# Attempt to load the five saved classification models.
try:

    # Load the saved pipelines without training them again.
    models = load_models()

    # Display a confirmation in the application.
    st.success("All five trained classification models loaded successfully.")

except Exception as error:

    # Display the error if any model cannot be loaded.
    st.error(f"Unable to load the saved models: {error}")

    # Stop the application to prevent invalid predictions.
    st.stop()


# ------------------------------------------------------------
# SECTION 4 - IDENTIFY REQUIRED INPUT FEATURES
# ------------------------------------------------------------

# Obtain the feature names recorded in the trained pipeline.
# This ensures that uploaded data matches the training schema.
reference_model = models["Logistic Regression"]

# Check whether the fitted pipeline contains feature names.
if not hasattr(reference_model, "feature_names_in_"):

    # Stop if the saved model does not provide the expected schema.
    st.error(
        "The saved pipeline does not contain input feature names. "
        "Please verify the model training pipeline."
    )

    st.stop()

# Convert the recorded feature names into a Python list.
required_features = list(reference_model.feature_names_in_)

# Verify that the saved pipeline expects exactly 16 input features.
if len(required_features) != 16:

    st.error(
        f"Expected 16 input features, but the model uses "
        f"{len(required_features)}."
    )

    st.stop()


# ------------------------------------------------------------
# SECTION 5 - SIDEBAR: TEST DATA UPLOAD
# ------------------------------------------------------------

# Display the sidebar heading.
st.sidebar.header("Test Data")

# Explain the required CSV format.
st.sidebar.write(
    "Upload a labeled CSV containing the 16 input features "
    "and the Class target column."
)

# Check whether the prepared sample test CSV exists.
if TEST_DATA_PATH.is_file():

    # Read the existing test CSV as binary data.
    sample_csv = TEST_DATA_PATH.read_bytes()

    # Provide a button to download the labeled test dataset.
    st.sidebar.download_button(
        label="Download Sample Test CSV",
        data=sample_csv,
        file_name="test_data.csv",
        mime="text/csv"
    )

else:

    # Warn the user when the sample CSV is unavailable.
    st.sidebar.warning("Sample test CSV is not available.")

# Create the CSV upload control.
uploaded_file = st.sidebar.file_uploader(
    "Upload Labeled Test CSV",
    type=["csv"]
)


# ------------------------------------------------------------
# SECTION 6 - SIDEBAR: CLASSIFIER SELECTION
# ------------------------------------------------------------

# Create a dropdown containing all five trained classifiers.
selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    options=list(MODEL_FILES.keys()),
    disabled=(uploaded_file is None)
)

# Display the number of loaded models in the sidebar.
st.sidebar.caption(
    f"Available trained models: {len(models)}"
)


# ------------------------------------------------------------
# SECTION 7 - VALIDATE UPLOADED TEST DATA
# ------------------------------------------------------------

# Continue only when a CSV file has been uploaded.
if uploaded_file is not None:

    # Attempt to read the uploaded file.
    try:

        # Load the CSV into a Pandas DataFrame.
        test_df = pd.read_csv(uploaded_file)

    except Exception as error:

        # Display an error when the file cannot be read.
        st.error(f"Unable to read the uploaded CSV: {error}")

        # Stop further execution.
        st.stop()

    # Check whether the uploaded CSV contains any observations.
    if test_df.empty:

        st.error("The uploaded CSV is empty.")

        st.stop()

    # Identify columns required by the trained model and evaluator.
    required_columns = required_features + [TARGET_COLUMN]

    # Find any required columns missing from the uploaded CSV.
    missing_columns = [
        column for column in required_columns
        if column not in test_df.columns
    ]

    # Reject the file if one or more required columns are missing.
    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        st.stop()

    # Reject duplicate column names to avoid ambiguous feature selection.
    if test_df.columns.duplicated().any():

        st.error("The uploaded CSV contains duplicate column names.")

        st.stop()

    # Extract the 16 input features in the trained model's order.
    X_uploaded = test_df[required_features].copy()

    # Extract the actual bean class labels.
    y_uploaded = test_df[TARGET_COLUMN].copy()

    # Convert all feature columns to numeric values.
    # Invalid text values become missing values for validation.
    X_uploaded = X_uploaded.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Check for missing or invalid feature values.
    if X_uploaded.isnull().any().any():

        st.error(
            "The uploaded CSV contains missing or non-numeric "
            "input feature values."
        )

        st.stop()

    # Check for missing target class labels.
    if y_uploaded.isnull().any():

        st.error("The Class column contains missing values.")

        st.stop()

    # Verify that all labels belong to the known training classes.
    known_classes = set(reference_model.classes_)

    # Identify any unknown target labels.
    unknown_classes = set(y_uploaded.unique()) - known_classes

    # Reject target labels that were not present during training.
    if unknown_classes:

        st.error(
            "Unknown bean classes found: "
            + ", ".join(map(str, unknown_classes))
        )

        st.stop()

    # Display a confirmation after all validation checks pass.
    st.success("Uploaded test CSV validated successfully.")

    # Display information about the uploaded dataset.
    col1, col2, col3 = st.columns(3)

    # Show the number of uploaded observations.
    col1.metric("Test Observations", len(test_df))

    # Show the number of input features.
    col2.metric("Input Features", len(required_features))

    # Show the number of target classes in the uploaded CSV.
    col3.metric("Classes in CSV", y_uploaded.nunique())


    # --------------------------------------------------------
    # SECTION 8 - GENERATE PREDICTIONS
    # --------------------------------------------------------

    # Retrieve the selected previously trained classifier.
    selected_model = models[selected_model_name]

    # Predict bean classes using the uploaded input features.
    # No model fitting or retraining occurs here.
    y_pred = selected_model.predict(X_uploaded)

    # Display the name of the selected classifier.
    st.subheader(f"Selected Model: {selected_model_name}")

    # Display confirmation that predictions were generated.
    st.success(
        f"Predictions generated for {len(y_pred)} observations."
    )


    # --------------------------------------------------------
    # SECTION 9 - PREPARE PREDICTION PREVIEW
    # --------------------------------------------------------

    # Create a copy of the uploaded dataset for displaying results.
    preview_df = test_df.copy()

    # Add the actual target labels for comparison.
    preview_df["Actual Class"] = y_uploaded

    # Add the predicted target labels.
    preview_df["Predicted Class"] = y_pred

    # Add a column indicating whether each prediction is correct.
    preview_df["Correct Prediction"] = (
        preview_df["Actual Class"] == preview_df["Predicted Class"]
    )

    # SECTION 10 - MODEL EVALUATION

    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        confusion_matrix,
        classification_report
    )
    import matplotlib.pyplot as plt
    import numpy as np

    # Use the actual target labels from the uploaded test CSV.
    y_true = y_uploaded

    # Use the predictions already generated by the selected model.
    # Reuse the predictions generated in Section 8.

    # Calculate multiclass evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC must use probabilities, not predicted class labels.
    # The probability columns follow the trained model's class order.
    y_prob = selected_model.predict_proba(X_uploaded)

    auc = roc_auc_score(
        y_true,
        y_prob,
        labels=selected_model.classes_,
        multi_class="ovr",
        average="weighted"
    )

    # Display all six mandatory metrics.
    st.subheader("Model Evaluation Metrics")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("Accuracy", f"{accuracy:.4f}")
    metric_col2.metric("AUC (OvR)", f"{auc:.4f}")
    metric_col3.metric("Precision", f"{precision:.4f}")

    metric_col4, metric_col5, metric_col6 = st.columns(3)

    metric_col4.metric("Recall", f"{recall:.4f}")
    metric_col5.metric("F1-score", f"{f1:.4f}")
    metric_col6.metric("MCC", f"{mcc:.4f}")

    st.caption(
        "Precision, Recall, F1 and AUC use weighted averaging. "
        "AUC uses One-vs-Rest classification probabilities."
    )

    # Generate the confusion matrix.
    st.subheader("Confusion Matrix")

    class_labels = selected_model.classes_

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))

    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    st.pyplot(fig, width=550)
    plt.close(fig)

    # Display the classification report as a table.
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.round(4))

    # ========================================================
    # SECTION 10 - FIVE-MODEL PERFORMANCE COMPARISON
    # ========================================================

    st.divider()
    st.subheader("Comparison of Five Classification Models")

    # Store the evaluation results for all trained classifiers.
    comparison_results = []

    # Evaluate each saved model on the same uploaded test data.
    for model_name, model in models.items():

        # Generate predicted class labels.
        comparison_pred = model.predict(X_uploaded)

        # Generate class probabilities for multiclass AUC.
        comparison_prob = model.predict_proba(X_uploaded)

        # Calculate multiclass AUC using One-vs-Rest.
        comparison_auc = roc_auc_score(
            y_uploaded,
            comparison_prob,
            labels=model.classes_,
            multi_class="ovr",
            average="weighted"
        )

        # Calculate all six mandatory evaluation metrics.
        comparison_results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(
                y_uploaded, comparison_pred
            ),
            "AUC": comparison_auc,
            "Precision": precision_score(
                y_uploaded, comparison_pred,
                average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_uploaded, comparison_pred,
                average="weighted", zero_division=0
            ),
            "F1-score": f1_score(
                y_uploaded, comparison_pred,
                average="weighted", zero_division=0
            ),
            "MCC": matthews_corrcoef(
                y_uploaded, comparison_pred
            )
        })

    # Convert the results into a comparison table.
    comparison_df = pd.DataFrame(comparison_results)

    # Sort models by accuracy, highest first.
    comparison_df = comparison_df.sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)

    # Display all six metrics rounded to four decimal places.
    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # Identify the model with the highest test accuracy.
    best_model_name = comparison_df.iloc[0]["Model"]
    best_accuracy = comparison_df.iloc[0]["Accuracy"]

    st.success(
        f"Highest Accuracy: {best_model_name} "
        f"({best_accuracy:.4f})"
    )

    # Explain the selection criterion clearly.
    st.caption(
        "Models are ranked by accuracy on the uploaded test "
        "dataset. AUC uses weighted One-vs-Rest probabilities. "
        "Precision, Recall and F1 use weighted averaging. "
        "All models are pre-trained and are not retrained here."
    )

    # ========================================================
    # SECTION 11 - DECISION TREE FEATURE IMPORTANCE
    # ========================================================
    # Display this visualization only for the Decision Tree.
    # Feature importance measures each feature's contribution
    # to reducing impurity across the trained decision tree.

    if selected_model_name == "Decision Tree":

        st.divider()
        st.subheader("Decision Tree - Feature Importance")

        # Retrieve the trained Decision Tree classifier.
        tree_classifier = selected_model.steps[-1][1]

        # Extract feature names and their importance scores.
        feature_names = list(selected_model.feature_names_in_)
        importance_scores = tree_classifier.feature_importances_

        # Create a table of feature importance values.
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_scores
        })

        # Select the ten most important bean measurements.
        top_features = (
            importance_df
            .sort_values("Importance", ascending=False)
            .head(10)
            .sort_values("Importance", ascending=True)
        )

        # Create a compact horizontal bar chart.
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            top_features["Feature"],
            top_features["Importance"],
            color="steelblue"
        )

        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Bean Measurement")
        ax.set_title("Top 10 Features Used by Decision Tree")

        fig.tight_layout()

        # Display the chart at a fixed width.
        st.pyplot(fig, width=550)
        plt.close(fig)

        st.caption(
            "Higher importance indicates a greater contribution "
            "to the Decision Tree's classification decisions."
        )


    # ========================================================
    # SECTION 12 - LOGISTIC REGRESSION COEFFICIENTS
    # ========================================================
    # Display this chart only for Logistic Regression.
    # Multiclass coefficients are summarized using the mean
    # absolute coefficient across all seven bean classes.

    if selected_model_name == "Logistic Regression":

        st.divider()
        st.subheader("Logistic Regression - Coefficient Importance")

        # Extract the trained classifier from the saved pipeline.
        lr_classifier = selected_model.steps[-1][1]

        # Retrieve the feature names and classifier coefficients.
        feature_names = list(selected_model.feature_names_in_)
        coefficients = lr_classifier.coef_

        # Calculate mean absolute coefficient across classes.
        importance_scores = abs(coefficients).mean(axis=0)

        # Create a feature-importance table.
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Absolute Coefficient": importance_scores
        })

        # Select the ten features with the largest coefficients.
        top_features = (
            importance_df
            .sort_values("Mean Absolute Coefficient", ascending=False)
            .head(10)
            .sort_values("Mean Absolute Coefficient")
        )

        # Create a compact horizontal bar chart.
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            top_features["Feature"],
            top_features["Mean Absolute Coefficient"],
            color="teal"
        )

        ax.set_xlabel("Mean Absolute Coefficient")
        ax.set_ylabel("Bean Measurement")
        ax.set_title("Top 10 Logistic Regression Coefficients")

        fig.tight_layout()

        st.pyplot(fig, width=550)
        plt.close(fig)

        st.caption(
            "Values represent the mean absolute coefficient "
            "across seven bean classes. Larger values indicate "
            "greater coefficient magnitude, not necessarily "
            "greater predictive importance."
        )



    # ========================================================
    # MODEL-SPECIFIC VISUALIZATION - KNN
    # ========================================================
    # Display this visualization only when KNN is selected.
    # Distances are calculated using the saved fitted scaler
    # and the existing training neighbors.

    if selected_model_name == "KNN":

        st.divider()
        st.subheader("KNN - Nearest Neighbor Distance Distribution")

        # Retrieve the fitted preprocessing and KNN steps.
        knn_scaler = selected_model.named_steps["scaler"]
        knn_classifier = selected_model.named_steps["classifier"]

        # Apply the scaler fitted on the training dataset.
        # Do not refit it using uploaded test observations.
        X_knn_scaled = knn_scaler.transform(X_uploaded)

        # Find the five closest training observations.
        neighbor_distances, _ = knn_classifier.kneighbors(
            X_knn_scaled,
            n_neighbors=5
        )

        # Extract the distance to the fifth nearest neighbor.
        fifth_distances = neighbor_distances[:, 4]

        # Calculate the median for the reference line.
        median_knn_distance = np.median(fifth_distances)

        # Create the distance distribution histogram.
        fig_knn, ax_knn = plt.subplots(figsize=(6, 4.5))

        ax_knn.hist(
            fifth_distances,
            bins=35,
            edgecolor="white",
            alpha=0.85
        )

        ax_knn.axvline(
            median_knn_distance,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median distance: {median_knn_distance:.3f}"
        )

        ax_knn.set_title(
            "KNN - Fifth Nearest Neighbor Distance Distribution"
        )
        ax_knn.set_xlabel(
            "Distance to Fifth Nearest Training Neighbor"
        )
        ax_knn.set_ylabel("Number of Test Observations")
        ax_knn.legend()

        plt.tight_layout()

        # Display the visualization in Streamlit.
        chart_col, empty_col = st.columns([1, 1.5])
        with chart_col:
            st.pyplot(fig_knn, use_container_width=True)
        plt.close(fig_knn)

        # Explain the chart.
        st.caption(
            "The histogram shows the distance from each uploaded "
            "test observation to its fifth nearest training neighbor. "
            "Smaller distances indicate greater proximity to training "
            "examples, not necessarily higher prediction accuracy."
        )



    # ========================================================
    # NAIVE BAYES - CLASS-WISE FEATURE DISTRIBUTION
    # ========================================================
    # Display this chart only when Naive Bayes is selected.
    # The plot compares actual bean classes in the uploaded
    # labeled test dataset using MajorAxisLength.

    if selected_model_name == "Naive Bayes":

        st.divider()
        st.subheader("Naive Bayes - Class-wise Feature Distribution")

        # Select the measurement to compare across bean classes.
        feature_name = "MajorAxisLength"

        # Sort the class labels for consistent display.
        bean_classes = sorted(y_uploaded.dropna().unique())

        # Collect the feature values for each actual class.
        class_values = [
            X_uploaded.loc[
                y_uploaded == bean_class,
                feature_name
            ].dropna().values
            for bean_class in bean_classes
        ]

        # Create the class-wise box plot.
        fig_nb, ax_nb = plt.subplots(figsize=(7, 4.5))

        ax_nb.boxplot(
            class_values,
            tick_labels=bean_classes,
            patch_artist=False
        )

        ax_nb.set_title(
            "Major Axis Length by Bean Class"
        )
        ax_nb.set_xlabel("Bean Class")
        ax_nb.set_ylabel("Major Axis Length")

        # Rotate labels to keep class names readable.
        plt.setp(
            ax_nb.get_xticklabels(),
            rotation=35,
            ha="right"
        )

        plt.tight_layout()

        # Keep the display compact, matching the KNN chart.
        nb_chart_col, nb_empty_col = st.columns([1, 1.5])

        with nb_chart_col:
            st.pyplot(fig_nb, use_container_width=True)

        plt.close(fig_nb)

        # Explain what the visualization represents.
        st.caption(
            "The box plots compare MajorAxisLength across "
            "the actual bean classes in the uploaded test CSV. "
            "Overlapping distributions indicate that this "
            "feature alone may not separate every class. "
            "The chart does not represent Naive Bayes "
            "prediction probabilities."
        )



    # ========================================================
    # SVM - PREDICTION CONFIDENCE DISTRIBUTION
    # ========================================================
    # Show the confidence distribution for the selected SVM.
    # Predictions use the existing pre-trained model.

    if selected_model_name == "SVM":

        st.divider()
        st.subheader("SVM - Prediction Confidence Distribution")

        # Generate probabilities for the seven bean classes.
        svm_probabilities = selected_model.predict_proba(X_uploaded)

        # Find the maximum class probability for each observation.
        svm_confidence = svm_probabilities.max(axis=1)

        # Calculate the median prediction confidence.
        median_confidence = float(np.median(svm_confidence))

        # Create a compact histogram.
        fig, ax = plt.subplots(figsize=(6.2, 3.8))

        ax.hist(
            svm_confidence,
            bins=25,
            color="steelblue",
            edgecolor="white"
        )

        # Mark the median confidence.
        ax.axvline(
            median_confidence,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_confidence:.3f}"
        )

        # Configure chart labels and axis limits.
        ax.set_title("SVM - Prediction Confidence Distribution")
        ax.set_xlabel("Maximum Predicted Class Probability")
        ax.set_ylabel("Number of Test Observations")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

        plt.tight_layout()

        # Display the chart at a compact width.
        st.pyplot(fig, width=550)
        plt.close(fig)

        # Explain what the chart represents.
        st.caption(
            "The histogram shows the highest predicted class "
            "probability for each uploaded test observation. "
            "Higher values indicate greater SVM prediction "
            "confidence, but do not guarantee correctness."
        )


    st.subheader("Test Data and Prediction Preview")

    # Show the first 20 observations and their predictions.
    st.dataframe(
        preview_df.head(20),
        use_container_width=True
    )

else:

    # Display instructions before the user uploads a file.
    st.info(
        "Upload the labeled test_data.csv file using the sidebar "
        "to begin model evaluation."
    )


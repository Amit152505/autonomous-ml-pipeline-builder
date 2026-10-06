"""
app.py
Autonomous ML Pipeline Builder - Streamlit Application.
Automated machine learning pipeline for tabular data.
"""

import io
import os
import traceback
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import streamlit as st

from src.data_profiler import DatasetProfiler
from src.problem_detector import ProblemDetector
from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_selector import ModelSelector
from src.explainability import ModelExplainer
from src.predictor import Predictor

# Page Configuration
st.set_page_config(
    page_title="Autonomous ML Pipeline Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
    .success-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .warning-badge {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "predictor" not in st.session_state:
    st.session_state.predictor = None


# Sidebar
with st.sidebar:
    st.title("🤖 AutoML Builder")
    st.caption("End-to-End AutoML Pipeline")
    st.divider()

    st.subheader("📁 Choose Dataset Source")
    data_source = st.radio(
        "Data Source",
        ["Use Sample Dataset", "Upload Custom CSV"],
        label_visibility="collapsed",
    )

    if data_source == "Use Sample Dataset":
        sample_choice = st.selectbox(
            "Select Sample Dataset",
            [
                "Customer Churn (Classification)",
                "House Prices (Regression)",
            ],
        )
        if st.button("📥 Load Selected Sample", use_container_width=True):
            if "Churn" in sample_choice:
                sample_path = "data/classification_sample.csv"
                st.session_state.dataset_name = "Customer Churn (Classification)"
                st.session_state.target_column = "churn"
            else:
                sample_path = "data/regression_sample.csv"
                st.session_state.dataset_name = "House Prices (Regression)"
                st.session_state.target_column = "price"

            if os.path.exists(sample_path):
                st.session_state.df = pd.read_csv(sample_path)
                st.session_state.pipeline_results = None
                st.session_state.predictor = None
                st.success(f"Loaded {st.session_state.dataset_name}!")
            else:
                st.error("Sample dataset file not found.")

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if uploaded_df.empty:
                    st.error("The uploaded CSV file is empty. Please upload a file with data.")
                elif len(uploaded_df) < 5:
                    st.error("The dataset has fewer than 5 rows. Supervised learning requires more data.")
                else:
                    if st.session_state.dataset_name != uploaded_file.name:
                        st.session_state.df = uploaded_df
                        st.session_state.dataset_name = uploaded_file.name
                        st.session_state.target_column = None
                        st.session_state.pipeline_results = None
                        st.session_state.predictor = None
                    st.success(f"Uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to read CSV: {str(e)}. Please verify file formatting.")

    st.divider()
    st.markdown(
        """
        **Pipeline Rules Enforced:**
        - ✅ Zero Data Leakage (Train/Test split before fit)
        - ✅ Median Imputation for numeric
        - ✅ Most-frequent Imputation for categorical
        - ✅ Standard Scaling & One-Hot Encoding
        - ✅ Baseline benchmarking (Dummy)
        - ✅ Pure deterministic ML logic
        """
    )


# Header
st.markdown('<div class="main-header">Autonomous ML Pipeline Builder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An automated, leakage-free machine learning pipeline that profiles data, detects problem types, trains multiple models, compares against baselines, and generates explainable predictions.</div>',
    unsafe_allow_html=True,
)

if st.session_state.df is None:
    # Empty State Welcome
    st.info("👈 Get started by selecting a sample dataset or uploading your own CSV file in the sidebar.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Sample 1: Customer Churn")
        st.markdown(
            """
            - **Problem Type**: Binary Classification
            - **Target**: `churn` (Yes / No)
            - **Features**: Age, Tenure, Monthly Charges, Contract Type, Tech Support
            - **Demonstrates**: Missing value imputation, categorical encoding, class balance check, Tree/XGBoost feature importances.
            """
        )
    with col2:
        st.markdown("### 🏠 Sample 2: House Prices")
        st.markdown(
            """
            - **Problem Type**: Continuous Regression
            - **Target**: `price` ($)
            - **Features**: Square Feet, Bedrooms, Bathrooms, Neighborhood, Year Built
            - **Demonstrates**: Scaling, continuous evaluation (RMSE, MAE, R²), actual vs. predicted diagnostics.
            """
        )
    st.stop()


# Dataset is loaded
df = st.session_state.df
profiler = DatasetProfiler(df)
basic_info = profiler.get_basic_info()
col_types = profiler.detect_column_types()

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. 📋 Data Overview",
    "2. 🔍 Data Profiler",
    "3. ⚙️ Setup & Detection",
    "4. 🚀 Train & Compare",
    "5. 🏆 Best Model & Explain",
    "6. 🔮 Predict & Export",
])

# ==============================================================================
# TAB 1: DATA OVERVIEW
# ==============================================================================
with tab1:
    st.subheader(f"Dataset Overview: {st.session_state.dataset_name}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", f"{basic_info['num_rows']:,}")
    m2.metric("Total Columns", f"{basic_info['num_columns']}")
    m3.metric("Duplicate Rows", f"{basic_info['duplicate_rows']}")
    m4.metric("Missing Cells", f"{basic_info['missing_cells']}")
    missing_pct_overall = (
        round((basic_info['missing_cells'] / basic_info['total_cells']) * 100, 2)
        if basic_info['total_cells'] > 0
        else 0
    )
    m5.metric("Missing Rate", f"{missing_pct_overall}%")

    st.markdown("#### First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Column Metadata & Data Types")
    col_meta = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
        "Missing Count": [int(df[c].isna().sum()) for c in df.columns],
    })
    st.dataframe(col_meta, use_container_width=True)


# ==============================================================================
# TAB 2: DATA PROFILER
# ==============================================================================
with tab2:
    st.subheader("Statistical Profiling & Diagnostics")

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("### Missing Value Breakdown")
        missing_df = profiler.get_missing_value_summary()
        st.dataframe(missing_df, use_container_width=True)

        if missing_df["Missing Values"].sum() > 0:
            missing_cols_only = missing_df[missing_df["Missing Values"] > 0]
            fig_miss = px.bar(
                missing_cols_only,
                x="Column",
                y="Missing %",
                color="Missing %",
                title="Missing Value Percentage by Column",
                color_continuous_scale="Reds",
            )
            fig_miss.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("🎉 Zero missing values detected in the dataset!")

    with col_p2:
        st.markdown("### Column Category Classification")
        st.markdown(f"**Numerical Columns ({len(col_types['numerical'])}):** `{', '.join(col_types['numerical']) or 'None'}`")
        st.markdown(f"**Categorical Columns ({len(col_types['categorical'])}):** `{', '.join(col_types['categorical']) or 'None'}`")

        if col_types["potential_ids"]:
            st.warning(
                f"⚠️ **Potential ID Column(s) Detected:** `{', '.join(col_types['potential_ids'])}`\n\n"
                f"These columns appear to be unique identifiers (keys/UUIDs) and may not be useful for prediction."
            )
        else:
            st.info("No high-cardinality ID columns detected.")

    st.divider()
    st.markdown("### Feature Distribution Explorer")
    selected_explore_col = st.selectbox("Select a feature to visualize its distribution:", df.columns)

    if selected_explore_col:
        is_num = selected_explore_col in col_types["numerical"]
        if is_num:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                fig_hist = px.histogram(
                    df,
                    x=selected_explore_col,
                    nbins=25,
                    title=f"Histogram: {selected_explore_col}",
                    color_discrete_sequence=["#3B82F6"],
                )
                fig_hist.update_layout(height=320)
                st.plotly_chart(fig_hist, use_container_width=True)
            with f_col2:
                fig_box = px.box(
                    df,
                    y=selected_explore_col,
                    title=f"Box Plot: {selected_explore_col}",
                    color_discrete_sequence=["#6366F1"],
                )
                fig_box.update_layout(height=320)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            cat_counts = df[selected_explore_col].value_counts().reset_index()
            cat_counts.columns = [selected_explore_col, "Count"]
            fig_bar = px.bar(
                cat_counts,
                x=selected_explore_col,
                y="Count",
                color=selected_explore_col,
                title=f"Value Counts: {selected_explore_col}",
            )
            fig_bar.update_layout(height=320)
            st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# TAB 3: SETUP & PROBLEM DETECTION
# ==============================================================================
with tab3:
    st.subheader("Target Selection & Problem Formulation")

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("#### 1. Select Target Column")
        # Default index
        col_list = list(df.columns)
        default_idx = 0
        if st.session_state.target_column and st.session_state.target_column in col_list:
            default_idx = col_list.index(st.session_state.target_column)
        elif len(col_list) > 1:
            default_idx = len(col_list) - 1

        selected_target = st.selectbox(
            "Which column do you want to predict?",
            col_list,
            index=default_idx,
        )
        st.session_state.target_column = selected_target

        # Validate target
        detector = ProblemDetector(df[selected_target])
        detection_res = detector.detect_problem_type()

        if not detection_res["is_valid"]:
            st.error(
                f"**Invalid Target Column:**\n\n"
                f"{detection_res['reason']}\n\n"
                f"Please choose a target with at least 2 distinct values."
            )
        else:
            st.success(f"Target `{selected_target}` is valid ({detection_res['unique_values_count']} unique values).")

        # Problem Type Inference Card
        st.markdown("#### 2. Detected Problem Type")
        detected_ptype = detection_res["problem_type"]
        st.info(
            f"**Recommended Task:** `{detected_ptype}`\n\n"
            f"**Reason:** {detection_res['reason']}"
        )

        # Allow user confirmation / toggle if ambiguous
        if detection_res.get("needs_confirmation"):
            user_ptype_override = st.radio(
                "Confirm problem formulation:",
                ["Classification", "Regression"],
                index=0 if detected_ptype == "Classification" else 1,
            )
            problem_type_final = user_ptype_override
        else:
            problem_type_final = detected_ptype

        # Class imbalance warning if classification
        if problem_type_final == "Classification" and detection_res.get("is_imbalanced"):
            st.warning(f"⚠️ {detection_res.get('imbalance_message')}")
            st.caption("The pipeline will automatically apply balanced class weights where supported.")

    with col_s2:
        st.markdown("#### 3. Target Distribution")
        if detection_res["is_valid"]:
            if problem_type_final == "Classification":
                dist = detector.get_class_distribution()
                dist_df = pd.DataFrame([
                    {"Class": k, "Count": v["count"], "Percentage": f"{v['percentage']}%"}
                    for k, v in dist.items()
                ])
                st.dataframe(dist_df, use_container_width=True)

                fig_t = px.bar(
                    dist_df,
                    x="Class",
                    y="Count",
                    color="Class",
                    text="Percentage",
                    title=f"Target Class Distribution: {selected_target}",
                )
                fig_t.update_layout(height=280)
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                fig_reg_t = px.histogram(
                    df,
                    x=selected_target,
                    nbins=25,
                    title=f"Target Distribution: {selected_target}",
                    color_discrete_sequence=["#10B981"],
                )
                fig_reg_t.update_layout(height=280)
                st.plotly_chart(fig_reg_t, use_container_width=True)

        st.markdown("#### 4. Feature Selection & Leakage Controls")
        # Option to drop ID columns
        potential_ids = col_types["potential_ids"]
        drop_ids_default = [c for c in potential_ids if c != selected_target]
        exclude_ids = st.checkbox(
            "Exclude high-cardinality ID columns from training (Recommended)",
            value=True if drop_ids_default else False,
            help="High-cardinality IDs lead to overfitting and contain no predictive signal.",
        )

        test_size = st.slider("Test Set Split Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        st.caption(f"Training split: {int((1-test_size)*100)}% | Test split: {int(test_size*100)}% (random_state=42)")


# ==============================================================================
# TAB 4: TRAIN & COMPARE
# ==============================================================================
with tab4:
    st.subheader("Autonomous Pipeline Execution")

    st.markdown(
        """
        Click the button below to trigger the autonomous training pipeline.
        The system will partition train/test sets, fit transformers strictly on training data to **prevent data leakage**,
        train a baseline plus candidate models, and rank performance on the unseen test set.
        """
    )

    ready_to_train = detection_res["is_valid"]

    if not ready_to_train:
        st.warning("Please configure a valid target column in the Setup tab first.")
    else:
        if st.button("🚀 Start Automated ML Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing Autonomous Pipeline..."):
                try:
                    # 1. Feature filtering
                    all_cols = list(df.columns)
                    features_to_use = [c for c in all_cols if c != selected_target]
                    dropped_cols = []

                    if exclude_ids and drop_ids_default:
                        dropped_cols = [c for c in drop_ids_default if c in features_to_use]
                        features_to_use = [c for c in features_to_use if c not in dropped_cols]

                    if not features_to_use:
                        st.error("No input features remaining after dropping target and IDs.")
                        st.stop()

                    X = df[features_to_use].copy()
                    y = df[selected_target].copy()

                    # Drop rows where target is NaN (cannot train/evaluate with unknown ground truth)
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]

                    # 2. Strict Train/Test Split BEFORE any preprocessing
                    stratify_arg = None
                    if problem_type_final == "Classification":
                        # Stratify only if each class has >= 2 samples
                        class_counts = y.value_counts()
                        if (class_counts >= 2).all():
                            stratify_arg = y

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42,
                        stratify=stratify_arg,
                    )

                    # Identify feature types on X
                    num_feats = [c for c in features_to_use if c in col_types["numerical"]]
                    cat_feats = [c for c in features_to_use if c in col_types["categorical"]]

                    # 3. Data Preprocessor (Fit strictly on TRAIN)
                    preprocessor = DataPreprocessor(
                        numerical_features=num_feats,
                        categorical_features=cat_feats,
                    )
                    X_train_trans = preprocessor.fit_transform(X_train)
                    X_test_trans = preprocessor.transform(X_test)

                    # Encode targets
                    y_train_trans, y_test_trans = preprocessor.prepare_target(
                        y_train=y_train,
                        y_test=y_test,
                        is_classification=(problem_type_final == "Classification"),
                    )

                    # 4. Train Models
                    trainer = ModelTrainer(
                        problem_type=problem_type_final,
                        is_imbalanced=detection_res.get("is_imbalanced", False),
                        random_state=42,
                    )

                    progress_placeholder = st.empty()

                    def on_progress(model_name, status):
                        progress_placeholder.info(f"⚡ Model: **{model_name}** - Status: `{status}`")

                    results = trainer.train_and_evaluate_all(
                        X_train=X_train_trans,
                        y_train=y_train_trans,
                        X_test=X_test_trans,
                        y_test=y_test_trans,
                        progress_callback=on_progress,
                    )
                    progress_placeholder.empty()

                    # 5. Select Best Model
                    selector = ModelSelector(
                        problem_type=problem_type_final,
                        evaluation_results=results,
                    )
                    best_info = selector.select_best_model()
                    best_model_name = best_info["best_model_name"]
                    best_model_obj = trainer.trained_models[best_model_name]

                    # 6. Feature metadata for predictor
                    feature_metadata = {}
                    for col in features_to_use:
                        if col in num_feats:
                            feature_metadata[col] = {
                                "type": "numerical",
                                "min": float(X_train[col].dropna().min()),
                                "max": float(X_train[col].dropna().max()),
                                "default": float(X_train[col].dropna().median()),
                            }
                        else:
                            cats = [str(x) for x in X_train[col].dropna().unique().tolist()]
                            feature_metadata[col] = {
                                "type": "categorical",
                                "options": cats,
                                "default": cats[0] if cats else "",
                            }

                    # 7. Initialize Predictor bundle
                    predictor = Predictor(
                        model=best_model_obj,
                        preprocessor=preprocessor,
                        target_encoder=preprocessor.target_encoder,
                        problem_type=problem_type_final,
                        feature_metadata=feature_metadata,
                    )

                    # 8. Save artifacts to models/
                    os.makedirs("models", exist_ok=True)
                    model_path = "models/model.joblib"
                    predictor.save(model_path)

                    # 9. Store in Session State
                    st.session_state.predictor = predictor
                    st.session_state.pipeline_results = {
                        "problem_type": problem_type_final,
                        "results": results,
                        "best_info": best_info,
                        "comparison_df": trainer.get_comparison_dataframe(),
                        "num_feats": num_feats,
                        "cat_feats": cat_feats,
                        "dropped_cols": dropped_cols,
                        "X_train_shape": X_train.shape,
                        "X_test_shape": X_test.shape,
                        "output_features": preprocessor.feature_names_out,
                        "y_test_trans": y_test_trans,
                        "detection_res": detection_res,
                        "best_model_name": best_model_name,
                    }

                    st.success("✅ Autonomous ML Pipeline execution completed successfully!")

                except Exception as e:
                    st.error(f"Unable to train the model: {str(e)}")
                    with st.expander("Diagnostic details"):
                        st.code(traceback.format_exc())

    # Display results if available
    if st.session_state.pipeline_results is not None:
        p_res = st.session_state.pipeline_results
        comp_df = p_res["comparison_df"]

        st.divider()
        st.markdown("### 📊 Model Performance Comparison")

        # Format comparison table
        st.dataframe(
            comp_df.style.highlight_max(
                axis=0,
                color="#DCFCE7",
                subset=[c for c in comp_df.columns if c in ["Accuracy", "Precision", "Recall", "F1", "R2"]],
            ).highlight_min(
                axis=0,
                color="#FEF9C3",
                subset=[c for c in comp_df.columns if c in ["MAE", "RMSE"]],
            ),
            use_container_width=True,
        )

        # Plotly Comparison Chart
        st.markdown("#### Performance Metric Chart")
        if p_res["problem_type"] == "Classification":
            metric_to_plot = st.selectbox("Metric to compare:", ["F1", "Accuracy", "Precision", "Recall"])
            fig_comp = px.bar(
                comp_df,
                x="Model",
                y=metric_to_plot,
                color="Model",
                text=metric_to_plot,
                title=f"Model Comparison: {metric_to_plot} Score",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_comp.update_layout(yaxis_range=[0, 1.05], height=380)
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            metric_to_plot = st.selectbox("Metric to compare:", ["RMSE", "MAE", "R2"])
            fig_comp = px.bar(
                comp_df,
                x="Model",
                y=metric_to_plot,
                color="Model",
                text=metric_to_plot,
                title=f"Model Comparison: {metric_to_plot}",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_comp.update_layout(height=380)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.info(
            "💡 **Why the Dummy Baseline Matters:**\n\n"
            "The Dummy baseline makes naive predictions (e.g. always predicting majority class or mean value) "
            "without looking at any features. Any viable machine learning model must comfortably beat the baseline. "
            "If an ML algorithm performs similarly to the baseline, the features may have weak predictive correlation."
        )


# ==============================================================================
# TAB 5: BEST MODEL & EXPLAINABILITY
# ==============================================================================
with tab5:
    st.subheader("Model Selection & Explainability Report")

    if st.session_state.pipeline_results is None:
        st.info("Run the training pipeline in Tab 4 to inspect the best model and explainability report.")
    else:
        p_res = st.session_state.pipeline_results
        best_info = p_res["best_info"]
        predictor = st.session_state.predictor

        col_b1, col_b2 = st.columns([1, 1])

        with col_b1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>🏆 Best Performing Model</h3>
                    <h2 style="color: #2563EB; margin: 4px 0;">{best_info['best_model_name']}</h2>
                    <p><b>Primary Metric:</b> {best_info['metric_name']} = <b>{best_info['best_metric_value']:.4f}</b></p>
                    <p style="color: #475569; font-size: 0.95rem;">{best_info['explanation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b2:
            st.markdown("### Metrics Summary")
            st.json(best_info["metrics"])

        st.divider()

        # Section 21: Autonomous Decision Summary
        st.markdown("### 📋 Autonomous Pipeline Decision Log")
        decision_text = ModelExplainer.generate_pipeline_summary(
            problem_type=p_res["problem_type"],
            problem_reason=p_res["detection_res"]["reason"],
            num_features=p_res["num_feats"],
            cat_features=p_res["cat_feats"],
            dropped_ids=p_res["dropped_cols"],
            best_model_name=best_info["best_model_name"],
            metric_name=best_info["metric_name"],
            best_score=best_info["best_metric_value"],
            best_reason=best_info["explanation"],
            models_trained=list(p_res["results"].keys()),
        )
        st.code(decision_text, language="text")

        st.divider()

        # Feature Importance
        st.markdown("### 🔍 Feature Importance")
        feat_importance_res = ModelExplainer.get_feature_importance(
            model=predictor.model,
            feature_names=p_res["output_features"],
            top_n=12,
        )

        if feat_importance_res["available"]:
            imp_df = feat_importance_res["importance_df"]
            f_col1, f_col2 = st.columns([1.2, 0.8])

            with f_col1:
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Top Features ({feat_importance_res['method']})",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=360)
                st.plotly_chart(fig_imp, use_container_width=True)

            with f_col2:
                st.markdown("**Top Feature Ranking**")
                display_imp_df = imp_df.copy()
                display_imp_df["Relative %"] = (display_imp_df["RelativeWeight"] * 100).round(1).astype(str) + "%"
                display_imp_df["Importance"] = display_imp_df["Importance"].round(4)
                st.dataframe(
                    display_imp_df[["Feature", "Importance", "Relative %"]],
                    use_container_width=True,
                )
                st.caption(feat_importance_res["explanation"])
        else:
            st.info(feat_importance_res["explanation"])

        st.divider()

        # Detailed Diagnostic Charts
        st.markdown("### 📈 Diagnostic Performance Plot")
        best_run_data = p_res["results"][p_res["best_model_name"]]

        if p_res["problem_type"] == "Classification":
            cm = best_run_data["metrics"].get("confusion_matrix")
            labels = best_run_data["metrics"].get("labels", ["0", "1"])
            if cm:
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=[f"Pred: {l}" for l in labels],
                    y=[f"True: {l}" for l in labels],
                    title=f"Confusion Matrix: {p_res['best_model_name']}",
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=360)
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            y_true_test = p_res["y_test_trans"]
            y_pred_test = best_run_data["y_pred"]
            diag_df = pd.DataFrame({"Actual": y_true_test, "Predicted": y_pred_test})
            fig_reg = px.scatter(
                diag_df,
                x="Actual",
                y="Predicted",
                title=f"Actual vs. Predicted Values ({p_res['best_model_name']})",
                trendline="ols",
                color_discrete_sequence=["#2563EB"],
            )
            # Add identity line
            min_val = min(diag_df["Actual"].min(), diag_df["Predicted"].min())
            max_val = max(diag_df["Actual"].max(), diag_df["Predicted"].max())
            fig_reg.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Ideal (1:1)",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig_reg.update_layout(height=400)
            st.plotly_chart(fig_reg, use_container_width=True)


# ==============================================================================
# TAB 6: PREDICT & EXPORT
# ==============================================================================
with tab6:
    st.subheader("Interactive Prediction & Model Export")

    if st.session_state.predictor is None:
        st.info("Train a model in Tab 4 first to enable interactive predictions and export.")
    else:
        predictor = st.session_state.predictor
        p_res = st.session_state.pipeline_results
        meta = predictor.feature_metadata

        st.markdown(
            f"Enter values below to make real-time predictions using the trained **{p_res['best_model_name']}**."
        )

        col_p_form, col_p_res = st.columns([1.1, 0.9])

        with col_p_form:
            st.markdown("#### Input Feature Values")
            user_inputs = {}

            # Build inputs responsive to columns
            cols = st.columns(2)
            i = 0
            for feat_name, feat_info in meta.items():
                with cols[i % 2]:
                    if feat_info["type"] == "numerical":
                        val = st.number_input(
                            label=feat_name,
                            min_value=feat_info["min"],
                            max_value=feat_info["max"],
                            value=feat_info["default"],
                            key=f"input_{feat_name}",
                        )
                        user_inputs[feat_name] = val
                    else:
                        options = feat_info.get("options", [])
                        default_val = feat_info.get("default", options[0] if options else "")
                        idx = options.index(default_val) if default_val in options else 0
                        val = st.selectbox(
                            label=feat_name,
                            options=options,
                            index=idx,
                            key=f"input_{feat_name}",
                        )
                        user_inputs[feat_name] = val
                i += 1

            predict_clicked = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)

        with col_p_res:
            st.markdown("#### Prediction Output")
            if predict_clicked:
                try:
                    res = predictor.predict(user_inputs)
                    pred_val = res["prediction"]

                    if p_res["problem_type"] == "Classification":
                        st.markdown(
                            f"""
                            <div class="metric-card" style="border-left: 5px solid #2563EB;">
                                <h4>Predicted Class</h4>
                                <h1 style="color: #2563EB; margin: 0;">{pred_val}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        probs = res.get("probabilities")
                        if probs:
                            st.markdown("##### Class Probabilities")
                            prob_df = pd.DataFrame([
                                {"Class": k, "Confidence": f"{v:.1f}%", "Probability": v}
                                for k, v in probs.items()
                            ])
                            st.dataframe(prob_df[["Class", "Confidence"]], use_container_width=True)
                            fig_p = px.bar(
                                prob_df,
                                x="Class",
                                y="Probability",
                                color="Class",
                                title="Confidence Distribution",
                            )
                            fig_p.update_layout(yaxis_range=[0, 100], height=240)
                            st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        st.markdown(
                            f"""
                            <div class="metric-card" style="border-left: 5px solid #10B981;">
                                <h4>Predicted Target Value</h4>
                                <h1 style="color: #10B981; margin: 0;">{pred_val:,.2f}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

        st.divider()

        # Section 20: Download Model
        st.markdown("### 💾 Export Serialized Model Bundle")
        st.markdown(
            "Download the complete end-to-end inference bundle (`model.joblib`), "
            "which encapsulates the fitted preprocessing `ColumnTransformer` alongside the trained estimator."
        )

        model_buffer = io.BytesIO()
        joblib.dump(
            {
                "model": predictor.model,
                "preprocessor": predictor.preprocessor,
                "target_encoder": predictor.target_encoder,
                "problem_type": predictor.problem_type,
                "feature_metadata": predictor.feature_metadata,
            },
            model_buffer,
        )
        model_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Trained Model Pipeline (model.joblib)",
            data=model_buffer,
            file_name="model.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Import models and helper functions
from models import LogisticRegression, CustomSVM, prepare_data, apply_resampling

# --- Dataset Loader ---
@st.cache_data
def load_dataset():
    """Load the actual diabetes dataset"""
    try:
        df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_dataset()

# --- Sidebar Navigation ---
st.sidebar.title("Diabetes Prediction Using Machine Learning")
section = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Diabetes Prediction"])

# --- Data Exploration Section ---
if section == "Data Exploration":
    st.title("Explore the Diabetes Health Indicators Dataset")

    if df is not None:
        st.subheader("Dataset Overview")
        st.write(f"Number of records: {df.shape[0]:,}")
        st.write(f"Number of features: {df.shape[1] - 1}")
        
        if st.checkbox("Show raw data"):
            st.dataframe(df.head(50))
        
        st.subheader("Feature Visualization")
        
        numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'Diabetes_binary']
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Diabetes_binary']
        
        all_cols = ['Diabetes_binary'] + numerical_cols + categorical_cols
        
        feature = st.selectbox("Select a feature to visualize:", all_cols)
        
        if feature == 'Diabetes_binary':
            st.write("Target Distribution (Diabetes)")
            diabetes_counts = df['Diabetes_binary'].value_counts()
            st.bar_chart(diabetes_counts)
            st.write(f"No Diabetes (0): {diabetes_counts.get(0, 0):,} ({diabetes_counts.get(0, 0)/len(df):.1%})")
            st.write(f"Diabetes (1): {diabetes_counts.get(1, 0):,} ({diabetes_counts.get(1, 0)/len(df):.1%})")
        else:
            st.write(f"Distribution of {feature}")
            
            if feature in numerical_cols:
                hist_values = np.histogram(df[feature], bins=20)[0]
                st.bar_chart(hist_values)
                
                st.write(f"Box Plot of {feature}")
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.boxplot(x=df[feature], ax=ax)
                ax.set_title(f'Box Plot: {feature}')
                st.pyplot(fig)
                
                stats = df[feature].describe()
                st.write(f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
            
            else:
                value_counts = df[feature].value_counts()
                st.bar_chart(value_counts)
    else:
        st.error("Dataset could not be loaded. Please check the file path and format.")

# --- Model Training Section ---
elif section == "Model Training":
    st.title("Train a Diabetes Prediction Model")
    st.write("Train a custom model using the diabetes dataset")

    if df is not None:
        model_type = st.selectbox("Model Type", ["Support Vector Machine", "Logistic Regression"])
        feature_selection = st.selectbox("Feature Selection", ["All Features", "Top Features by Correlation"])
        
        num_top_features = None
        if feature_selection == "Top Features by Correlation":
            target_col = 'Diabetes_binary'
            corr_matrix = df.corr()
            target_corr = corr_matrix[target_col].drop(target_col).abs()
            sorted_features = target_corr.sort_values(ascending=False)
            
            with st.expander("View Feature Correlations with Diabetes"):
                st.bar_chart(sorted_features)
            
            max_features = len(sorted_features)
            num_top_features = st.slider("Number of Top Features", min_value=1, max_value=max_features, value=10)
            
            top_features = sorted_features.head(num_top_features).index.tolist()
            st.write(f"Selected top {num_top_features} features:")
            st.write(", ".join(top_features))
        
        resampling = st.selectbox("Resampling Strategy", ["None", "SMOTE", "SMOTE + Undersampling", "ENN"])
        num_iterations = st.slider("Number of Iterations (Epochs)", min_value=100, max_value=5000, value=2000, step=100)
        
        st.subheader("Learning Rate")
        learning_rate_method = st.radio(
            "Learning Rate Selection Method", 
            ["Choose from presets", "Enter custom value"],
            horizontal=True
        )
        
        if learning_rate_method == "Choose from presets":
            lr_options = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
            learning_rate = st.select_slider(
                "Select Learning Rate",
                options=lr_options,
                value=0.001,
                format_func=lambda x: f"{x:.5f}"
            )
        else:
            learning_rate = st.number_input(
                "Enter Custom Learning Rate", 
                min_value=0.000001, 
                max_value=1.0, 
                value=0.001, 
                step=0.00001,
                format="%.6f"
            )
            
        st.write(f"Selected learning rate: **{learning_rate:.6f}**")
        
        lambda_param = 0.001
        if model_type == "Support Vector Machine":
            st.subheader("SVM Regularization Parameter")
            lambda_method = st.radio(
                "Lambda Selection Method", 
                ["Choose from presets", "Enter custom value"],
                horizontal=True
            )
            
            if lambda_method == "Choose from presets":
                lambda_options = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
                lambda_param = st.select_slider(
                    "Select Lambda",
                    options=lambda_options,
                    value=0.001,
                    format_func=lambda x: f"{x:.5f}"
                )
            else:
                lambda_param = st.number_input(
                    "Enter Custom Lambda", 
                    min_value=0.000001, 
                    max_value=1.0, 
                    value=0.001, 
                    step=0.00001,
                    format="%.6f"
                )
                
            st.write(f"Selected lambda: **{lambda_param:.6f}**")
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                if feature_selection == "Top Features by Correlation":
                    target_col = 'Diabetes_binary'
                    corr_matrix = df.corr()
                    target_corr = corr_matrix[target_col].drop(target_col).abs()
                    top_features = target_corr.sort_values(ascending=False).head(num_top_features).index.tolist()
                    
                    st.write(f"Using top {num_top_features} features by correlation with target:")
                    st.write(", ".join(top_features))
                    
                    X_train, X_test, y_train, y_test = prepare_data(df, feature_cols=top_features)
                else:
                    X_train, X_test, y_train, y_test = prepare_data(df)
                
                X_train, y_train, resampling_description = apply_resampling(X_train, y_train, resampling)
                st.write(resampling_description)
                
                progress_bar = st.progress(0)
                loss_chart_placeholder = st.empty()
                current_loss_placeholder = st.empty()
                
                fig, ax = plt.subplots(figsize=(10, 4))
                line, = ax.plot([], [], 'b-' if model_type == "Logistic Regression" else 'r-')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss')
                ax.set_title(f'Training Progress - {"Binary Cross-Entropy" if model_type == "Logistic Regression" else "Hinge"} Loss')
                ax.grid(True)
                
                def update_training_ui(iteration, progress, loss):
                    progress_bar.progress(progress)
                    current_loss_placeholder.write(f"Current Loss: {loss:.6f}")
                    
                    iterations = [i * 100 for i in range(len(loss_history))]
                    line.set_data(iterations, loss_history)
                    
                    if len(iterations) > 0:
                        ax.set_xlim(0, iterations[-1] + 100)
                        if len(loss_history) > 0:
                            min_loss = min(loss_history)
                            max_loss = max(loss_history)
                            y_margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
                            ax.set_ylim(min_loss - y_margin, max_loss + y_margin)
                    
                    loss_chart_placeholder.pyplot(fig)
                
                st.write(f"Training {model_type} with {num_iterations} iterations...")
                st.write("**Watch training progress in real-time below:**")
                
                loss_history = []
                
                def callback_with_history(iteration, progress, loss):
                    loss_history.append(loss)
                    update_training_ui(iteration, progress, loss)
                
                if model_type == "Logistic Regression":
                    model = LogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)
                    model.fit(X_train, y_train, callback=callback_with_history)
                    y_pred = model.predict(X_test)
                    loss_history = model.loss_history
                    
                else:  # Support Vector Machine
                    model = CustomSVM(
                        learning_rate=learning_rate,
                        num_iterations=num_iterations,
                        lambda_param=lambda_param,
                        weighted_loss=False
                    )
                    model.fit(X_train, y_train, callback=callback_with_history)
                    y_pred = model.predict(X_test)
                    loss_history = model.loss_history
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, pos_label=1)
                recall = recall_score(y_test, y_pred, pos_label=1)
                precision = precision_score(y_test, y_pred, pos_label=1)
                
                # Calculate class-specific metrics
                diabetes_precision = precision_score(y_test, y_pred, pos_label=1)
                diabetes_recall = recall_score(y_test, y_pred, pos_label=1)
                diabetes_f1 = f1_score(y_test, y_pred, pos_label=1)
                
                non_diabetes_precision = precision_score(y_test, y_pred, pos_label=0)
                non_diabetes_recall = recall_score(y_test, y_pred, pos_label=0)
                non_diabetes_f1 = f1_score(y_test, y_pred, pos_label=0)
                
                st.success(f"Training completed successfully!")
                st.subheader("Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1-score", f"{f1:.2%}")
                
                st.subheader("Detailed Performance Metrics")
                
                metrics_data = {
                    'Class': ['Non-Diabetes (0)', 'Diabetes (1)', 'Overall'],
                    'Precision': [f"{non_diabetes_precision:.2%}", f"{diabetes_precision:.2%}", f"{precision:.2%}"],
                    'Recall': [f"{non_diabetes_recall:.2%}", f"{diabetes_recall:.2%}", f"{recall:.2%}"],
                    'F1-Score': [f"{non_diabetes_f1:.2%}", f"{diabetes_f1:.2%}", f"{f1:.2%}"],
                    'Support': [f"{sum(y_test == 0)}", f"{sum(y_test == 1)}", f"{len(y_test)}"]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.table(metrics_df)
                
                st.write("""
                **Interpretation Guide:**
                - **Precision**: When the model predicts a class, how often is it correct?
                - **Recall**: Out of all actual instances of a class, how many did the model correctly identify?
                - **F1-Score**: Harmonic mean of precision and recall (balance between the two)
                - **Support**: Number of actual occurrences of the class in the test set
                """)
                
                cm = confusion_matrix(y_test, y_pred)
                
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                ax.set_title('Confusion Matrix')
                ax.set_xticklabels(['Non-Diabetes', 'Diabetes'])
                ax.set_yticklabels(['Non-Diabetes', 'Diabetes'])
                
                st.pyplot(fig)
                
                st.markdown("""
                <div style="background-color: rgba(255, 165, 0, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 20px;">
                <h4>Confusion Matrix Explanation:</h4>
                <ul>
                <li><strong>TN (top-left):</strong> True Negatives - Correctly predicted non-diabetic patients</li>
                <li><strong>FP (top-right):</strong> False Positives - Non-diabetic patients incorrectly predicted as diabetic</li>
                <li><strong>FN (bottom-left):</strong> False Negatives - Diabetic patients incorrectly predicted as non-diabetic</li>
                <li><strong>TP (bottom-right):</strong> True Positives - Correctly predicted diabetic patients</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Training Loss Curve")
                fig, ax = plt.subplots(figsize=(10, 5))
                iterations = [i for i in range(len(loss_history))]
                ax.plot(iterations, loss_history)
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss')
                title_prefix = "Binary Cross-Entropy" if model_type == "Logistic Regression" else "Hinge"
                ax.set_title(f'{title_prefix} Loss During Training')
                ax.grid(True)
                st.pyplot(fig)
                
                if model_type == "Support Vector Machine":
                    st.subheader("Threshold Analysis")
                    
                    non_diabetes_precision_list = []
                    non_diabetes_recall_list = []
                    diabetes_precision_list = []
                    diabetes_recall_list = []
                    threshold_list = []
                    f1_score_list = []
                    accuracy_list = []
                    thresholds = np.arange(-1.0, 1.01, 0.1)

                    for t in thresholds:
                        pred = model.predict(X_test, threshold=t)

                        non_diabetes_precision_list.append(
                            precision_score(y_test, pred, pos_label=0, zero_division=0)
                        )
                        non_diabetes_recall_list.append(
                            recall_score(y_test, pred, pos_label=0, zero_division=0)
                        )

                        diabetes_precision_list.append(
                            precision_score(y_test, pred, pos_label=1, zero_division=0)
                        )
                        diabetes_recall_list.append(
                            recall_score(y_test, pred, pos_label=1, zero_division=0)
                        )

                        f1_score_list.append(
                            f1_score(y_test, pred, pos_label=1, zero_division=0)
                        )

                        accuracy_list.append(
                            accuracy_score(y_test, pred)
                        )

                        threshold_list.append(t)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(threshold_list, non_diabetes_precision_list, label='Non-Diabetes Precision', color='blue')
                    ax.plot(threshold_list, non_diabetes_recall_list, label='Non-Diabetes Recall', color='lightblue')
                    ax.plot(threshold_list, diabetes_precision_list, label='Diabetes Precision', color='red')
                    ax.plot(threshold_list, diabetes_recall_list, label='Diabetes Recall', color='pink')
                    ax.plot(threshold_list, f1_score_list, label='F1 Score', color='green')
                    ax.set_xlabel('Threshold')
                    ax.set_ylabel('Precision/Recall')
                    ax.legend()
                    ax.set_title(f'Precision and Recall vs Threshold ({model_type} with {resampling})')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
    else:
        st.error("Dataset could not be loaded. Please check the file path and format.")

# --- Prediction Section ---
elif section == "Diabetes Prediction":
    st.title("Diabetes Risk Prediction")
    
    st.write("This tool uses pre-trained machine learning models to predict diabetes risk based on health indicators.")
    
    top_features = ['GenHlth', 'HighBP', 'DiffWalk', 'BMI', 'HighChol', 'Age', 
                     'HeartDiseaseorAttack', 'PhysHlth', 'Income', 'Education']
    
    st.info(f"This prediction uses only the top 10 most important features: {', '.join(top_features)}")
    
    import pickle
    import os
    
    model_choice = st.selectbox("Select prediction model", ["Support Vector Machine", "Logistic Regression"])
    
    model_file = "lr_model.pkl" if model_choice == "Logistic Regression" else "svm_model.pkl"
    
    if not os.path.exists(model_file):
        st.error(f"Model file {model_file} not found. Please train and save the model first.")
    elif not os.path.exists("scalers.pkl"):
        st.error("Scalers file not found. Please train and save the model first.")
    else:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            with open("scalers.pkl", 'rb') as f:
                scalers = pickle.load(f)
                
            st.subheader("Enter your health information:")
            
            col1, col2 = st.columns(2)
            
            feature_values = {}
            
            with col1:
                if 'HighBP' in top_features:
                    high_bp = st.selectbox("High Blood Pressure (HighBP)", 
                                        options=["No high BP", "High BP"],
                                        format_func=lambda x: x)
                    feature_values['HighBP'] = 1 if high_bp == "High BP" else 0
                
                if 'HighChol' in top_features:
                    high_chol = st.selectbox("High Cholesterol (HighChol)", 
                                        options=["No high cholesterol", "High cholesterol"],
                                        format_func=lambda x: x)
                    feature_values['HighChol'] = 1 if high_chol == "High cholesterol" else 0
                
                if 'BMI' in top_features:
                    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
                    feature_values['BMI'] = bmi
                
                if 'HeartDiseaseorAttack' in top_features:
                    heart_disease = st.selectbox("Heart Disease or Attack", 
                                            options=["No", "Yes"],
                                            format_func=lambda x: x)
                    feature_values['HeartDiseaseorAttack'] = 1 if heart_disease == "Yes" else 0
                
                if 'PhysHlth' in top_features:
                    phys_hlth = st.number_input("Days of Poor Physical Health in past 30 days", 
                                           0, 30, 0,
                                           help="For how many days during the past 30 days was your physical health not good?")
                    feature_values['PhysHlth'] = phys_hlth
            
            with col2:
                if 'GenHlth' in top_features:
                    gen_hlth_options = ["Excellent", "Very good", "Good", "Fair", "Poor"]
                    gen_hlth = st.selectbox("General Health (GenHlth)", 
                                       options=gen_hlth_options,
                                       format_func=lambda x: x)
                    feature_values['GenHlth'] = gen_hlth_options.index(gen_hlth) + 1  # Convert to 1-5 scale
                
                if 'DiffWalk' in top_features:
                    diff_walk = st.selectbox("Difficulty Walking or Climbing Stairs", 
                                        options=["No", "Yes"],
                                        format_func=lambda x: x)
                    feature_values['DiffWalk'] = 1 if diff_walk == "Yes" else 0
                
                if 'Age' in top_features:
                    age_options = [
                        "18 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44",
                        "45 to 49", "50 to 54", "55 to 59", "60 to 64", "65 to 69",
                        "70 to 74", "75 to 79", "80 or older", "Don't know/Refused/Missing"
                    ]
                    age = st.selectbox("Age Category", 
                                  options=age_options,
                                  index=8,  # Default to 60-64
                                  format_func=lambda x: x)
                    feature_values['Age'] = age_options.index(age) + 1  # Convert to 1-14 scale
                
                if 'Education' in top_features:
                    education_options = [
                        "Never attended school or only kindergarten",
                        "Grades 1 through 8 (Elementary)",
                        "Grades 9 through 11 (Some high school)",
                        "Grade 12 or GED (High school graduate)",
                        "College 1 year to 3 years (Some college or technical school)",
                        "College 4 years or more (College graduate)"
                    ]
                    education = st.selectbox("Education Level", 
                                        options=education_options,
                                        format_func=lambda x: x)
                    feature_values['Education'] = education_options.index(education) + 1  # Convert to 1-6 scale
                
                if 'Income' in top_features:
                    income_options = [
                        "Less than $15,000",
                        "$15,000 to less than $25,000",
                        "$25,000 to less than $35,000",
                        "$35,000 to less than $50,000",
                        "$50,000 or more",
                        "Don't know/Not sure/Missing"
                    ]
                    income = st.selectbox("Income Level", 
                                     options=income_options,
                                     format_func=lambda x: x)
                    feature_values['Income'] = income_options.index(income) + 1  # Convert to 1-6 scale
            
            st.info("Note: All features will be encoded to their numerical values according to the dataset requirements.")
            
            if st.button("Predict Diabetes Risk"):
                all_features_dict = {
                    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 25.0, "Smoker": 0, 
                    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, 
                    "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, 
                    "GenHlth": 3, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, 
                    "Age": 9, "Education": 4, "Income": 5
                }
                
                all_features_dict.update(feature_values)
                
                feature_names = [
                    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
                ]
                
                features = [all_features_dict[name] for name in feature_names]
                
                input_df = pd.DataFrame([features], columns=feature_names)
                
                with st.expander("View All Encoded Features"):
                    st.write("Features highlighted in green are used for prediction:")
                    
                    def highlight_top_features(val, column_name):
                        return 'background-color: #c6ecc6' if column_name in top_features else ''
                    
                    styled_df = input_df.style.apply(lambda _: pd.DataFrame([highlight_top_features(_, col) for col in input_df.columns], 
                                                                         index=input_df.columns).T, axis=None)
                    st.dataframe(styled_df)
                
                input_features = np.array(features).reshape(1, -1)
                
                try:
                    robust_scaler = scalers["robust_scaler"]
                    standard_scaler = scalers["standard_scaler"]
                    
                    normalized_features = input_features.copy()
                    
                    robust_indices = [3, 14, 15]  # bmi, ment_hlth, phys_hlth
                    robust_features = input_features[:, robust_indices]
                    normalized_features[:, robust_indices] = robust_scaler.transform(robust_features)
                    
                    standard_indices = [13, 18, 19, 20]  # gen_hlth, age, education, income
                    standard_features = input_features[:, standard_indices]
                    normalized_features[:, standard_indices] = standard_scaler.transform(standard_features)
                    
                    with st.expander("View Normalized Features"):
                        normalized_df = pd.DataFrame(normalized_features, columns=feature_names)
                        st.dataframe(normalized_df)
                    
                    indices_of_top_features = [feature_names.index(feature) for feature in top_features]
                    top_features_data = normalized_features[:, indices_of_top_features]
                    
                    with st.expander("View Top 10 Features Used for Prediction"):
                        top_df = pd.DataFrame(top_features_data, columns=top_features)
                        st.dataframe(top_df)
                    
                except Exception as e:
                    st.error(f"Error during feature normalization: {str(e)}")
                    st.write("Please check the structure of the scalers file and ensure it matches expectations.")
                    raise
                
                if model_choice == "Logistic Regression":
                    prediction = model.predict(top_features_data)
                    
                    raw_output = None
                    probability = None
                    raw_score_source = "approximated"
                    logit_score = None
                    
                    if hasattr(model, 'predict_probability'):
                        probability = model.predict_probability(top_features_data)[0]
                        raw_output = probability
                        raw_score_source = "predict_probability (sigmoid output)"
                        
                        logit_score = np.log(probability / (1 - probability)) if 0 < probability < 1 else 0
                    
                    elif hasattr(model, 'predict_proba'):
                        proba_output = model.predict_proba(top_features_data)
                        probability = proba_output[0][1]  # Probability of class 1
                        raw_output = np.log(probability / (1 - probability)) if 0 < probability < 1 else 0
                        raw_score_source = "predict_proba"
                        
                    elif hasattr(model, 'decision_function'):
                        raw_output = model.decision_function(top_features_data)[0]
                        probability = 1 / (1 + np.exp(-raw_output))
                        raw_score_source = "decision_function"
                    
                    elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                        raw_output = np.dot(top_features_data, model.coef_.T)[0][0] + model.intercept_[0]
                        probability = 1 / (1 + np.exp(-raw_output))
                        raw_score_source = "coefficients"
                
                else:  # Support Vector Machine
                    prediction = model.predict(top_features_data)
                    raw_output = None
                    probability = None
                    raw_score_source = "approximated"
                    
                    if hasattr(model, 'predict_score'):
                        raw_score = model.predict_score(top_features_data)
                        if isinstance(raw_score, np.ndarray):
                            raw_output = raw_score[0]
                        else:
                            raw_output = raw_score
                        
                        probability = 1 / (1 + np.exp(-raw_output))
                        raw_score_source = "predict_score"
                        
                    elif hasattr(model, 'predict_probability'):
                        probability = model.predict_probability(top_features_data)[0]
                        raw_output = probability
                        raw_score_source = "predict_probability"
                
                risk_thresholds = [0.4, 0.7]
                risk_colors = ['green', 'orange', 'red']

                st.subheader("Prediction Results")

                if probability is not None:
                    if probability >= risk_thresholds[1]:  # High risk
                        st.warning("**Prediction: High risk for diabetes**", icon="⚠️")
                        risk_level = "High"
                        risk_color = risk_colors[2]
                    elif probability >= risk_thresholds[0]:  # Medium risk
                        st.warning("**Prediction: Medium risk for diabetes**", icon="ℹ️")
                        risk_level = "Medium"
                        risk_color = risk_colors[1]
                    else:  # Low risk
                        st.success("**Prediction: Low risk for diabetes**", icon="✅")
                        risk_level = "Low"
                        risk_color = risk_colors[0]
                else:
                    if prediction[0] == 1:
                        st.warning("**Prediction: At risk for diabetes**", icon="⚠️")
                        risk_level = "Unknown"
                        risk_color = "gray"
                    else:
                        st.success("**Prediction: Not at risk for diabetes**", icon="✅")
                        risk_level = "Unknown"
                        risk_color = "gray"

                st.markdown("### Model Technical Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Raw Model Output:**")
                    if raw_output is not None:
                        st.info(f"{raw_output:.6f}")
                    else:
                        st.info("No raw output available")
                    
                    st.markdown(f"**Model Type:** {model_choice}")

                with col2:
                    st.markdown("**Confidence of Diabetes:**")
                    if probability is not None:
                        st.info(f"{probability:.4f} ({probability:.2%}) - **{risk_level} Risk**")
                    else:
                        approx_prob = 0.99 if prediction[0] == 1 else 0.01
                        st.info(f"{approx_prob:.4f} ({approx_prob:.2%}) (approximated)")
                
                if probability is not None:
                    st.write(f"Confidence of diabetes: **{probability:.2%}**")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    thresholds = [0] + risk_thresholds + [1.0]

                    ax.barh([0.6], [1.0], color='lightgray', height=0.2)

                    for i in range(len(thresholds)-1):
                        width = thresholds[i+1] - thresholds[i]
                        ax.barh([0.6], [width], left=[thresholds[i]], color=risk_colors[i], height=0.2)

                    ax.axvline(x=probability, color='black', linestyle='-', linewidth=2)
                    ax.text(probability, 0.6, f"{probability:.2%}", ha='center', va='bottom', fontweight='bold')

                    risk_text_y = 0.3
                    ax.text(0.2, risk_text_y, "Low Risk", ha='center', va='center', color=risk_colors[0], fontweight='bold')
                    ax.text(0.55, risk_text_y, "Medium Risk", ha='center', va='center', color=risk_colors[1], fontweight='bold')
                    ax.text(0.85, risk_text_y, "High Risk", ha='center', va='center', color=risk_colors[2], fontweight='bold')

                    for tick in thresholds:
                        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.5, ymin=0.55, ymax=0.65)
                        if tick > 0 and tick < 1:
                            ax.text(tick, 0.7, f"{tick:.1f}", ha='center', va='bottom', fontsize=8, color='gray')

                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(0, 1)
                    ax.axis('off')

                    st.pyplot(fig)
                
                st.info("Disclaimer: This prediction is based on a machine learning model and should not be considered medical advice. Please consult a healthcare professional for proper diagnosis and advice.")
                
        except Exception as e:
            st.error(f"Error loading model or making prediction: {str(e)}")
            st.write("Please ensure the model files are correctly saved and accessible.")


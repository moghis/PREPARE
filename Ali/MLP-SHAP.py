import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score, roc_curve

# Load the data
file_path = '/home/data3/Ali/Code/Moghis/Train-data-preprocess-2.xlsx'  

data = pd.read_excel(file_path)

# Define features and target label
features = [
    'uid', 'age_03', 'urban_03', 'married_03', 'n_mar_03', 'edu_gru_03', 'n_living_child_03',
    'migration_03', 'glob_hlth_03', 'adl_dress_03', 'adl_walk_03', 'adl_bath_03', 'adl_eat_03',
    'adl_bed_03', 'adl_toilet_03', 'n_adl_03', 'iadl_money_03', 'iadl_meds_03', 'iadl_shop_03',
    'iadl_meals_03', 'n_iadl_03', 'depressed_03', 'hard_03', 'restless_03', 'happy_03', 'lonely_03',
    'enjoy_03', 'sad_03', 'tired_03', 'energetic_03', 'n_depr_03', 'cesd_depressed_03',
    'hypertension_03', 'diabetes_03', 'resp_ill_03', 'arthritis_03', 'hrt_attack_03', 'stroke_03',
    'cancer_03', 'n_illnesses_03', 'exer_3xwk_03', 'alcohol_03', 'tobacco_03', 'test_chol_03',
    'test_tuber_03', 'test_diab_03', 'test_pres_03', 'hosp_03', 'visit_med_03', 'out_proc_03',
    'visit_dental_03', 'imss_03', 'issste_03', 'pem_def_mar_03', 'insur_private_03', 'insur_other_03',
    'insured_03', 'decis_personal_03', 'employment_03', 'age_12', 'urban_12', 'married_12',
    'n_mar_12', 'edu_gru_12', 'n_living_child_12', 'migration_12', 'glob_hlth_12', 'adl_dress_12',
    'adl_walk_12', 'adl_bath_12', 'adl_eat_12', 'adl_bed_12', 'adl_toilet_12', 'n_adl_12',
    'iadl_money_12', 'iadl_meds_12', 'iadl_shop_12', 'iadl_meals_12', 'n_iadl_12', 'depressed_12',
    'hard_12', 'restless_12', 'happy_12', 'lonely_12', 'enjoy_12', 'sad_12', 'tired_12',
    'energetic_12', 'n_depr_12', 'cesd_depressed_12', 'hypertension_12', 'diabetes_12',
    'resp_ill_12', 'arthritis_12', 'hrt_attack_12', 'stroke_12', 'cancer_12', 'n_illnesses_12',
    'bmi_12', 'exer_3xwk_12', 'alcohol_12', 'tobacco_12', 'test_chol_12', 'test_tuber_12',
    'test_diab_12', 'test_pres_12', 'hosp_12', 'visit_med_12', 'out_proc_12', 'visit_dental_12',
    'imss_12', 'issste_12', 'pem_def_mar_12', 'insur_private_12', 'insur_other_12', 'insured_12',
    'decis_famil_12', 'decis_personal_12', 'employment_12', 'vax_flu_12', 'vax_pneu_12', 'seg_pop_12',
    'care_adult_12', 'care_child_12', 'volunteer_12', 'attends_class_12', 'attends_club_12',
    'reads_12', 'games_12', 'table_games_12', 'comms_tel_comp_12', 'act_mant_12', 'tv_12',
    'sewing_12', 'satis_ideal_12', 'satis_excel_12', 'satis_fine_12', 'cosas_imp_12',
    'wouldnt_change_12', 'memory_12', 'ragender', 'rameduc_m', 'rafeduc_m', 'sgender_03',
    'rearnings_03', 'searnings_03', 'hincome_03', 'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03',
    'hinc_cap_03', 'rinc_pension_03', 'sinc_pension_03', 'rrelgimp_03', 'sgender_12', 'rjlocc_m_12',
    'rearnings_12', 'searnings_12', 'hincome_12', 'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12',
    'hinc_cap_12', 'rinc_pension_12', 'sinc_pension_12', 'rrelgimp_12', 'rrfcntx_m_12',
    'rsocact_m_12', 'rrelgwk_12', 'a34_12', 'j11_12', 'year', 'composite_score', 'hincome_change',
    'niadl_change', 'adl_change', 'depr_change', 'glob_hlth_change', 'edu_gru_change',
    'illnesses_change'
    ]

target = 'composite_score'

# Print dataset size
print(f"Total number of samples: {len(data)}")

# Data preprocessing
data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[features] = data[features].fillna(data[features].mean())
X = data[features]
y = data[target]

# Convert target to binary classes (0 and 0.5)
# y = (y == 0.5).astype(int)

# Print class distribution
print("\nClass distribution:")
print(y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames with feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)

# Initialize and train MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = mlp.predict(X_test_scaled)
y_pred_proba = mlp.predict_proba(X_test_scaled)


# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"\nAUC: {auc:.4f}")


# Print classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 0.5']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

def analyze_feature_importance_and_shap(model, X_train_df, X_test_df, feature_names):
    """
    Analyze feature importance using SHAP values for MLP Classifier with fixed shape handling
    """
    print("\nCalculating SHAP values...")
    
    # Create DeepExplainer for MLP
    background = shap.sample(X_train_df, 100)
    
    def predict_proba_wrapper(X):
        return model.predict_proba(X.astype(np.float32))
    
    explainer = shap.KernelExplainer(predict_proba_wrapper, background)
    
    # Calculate SHAP values for a subset of test data
    n_samples = min(50, len(X_test_df))  # Reduced number of samples for computation
    test_samples = X_test_df.iloc[:n_samples]
    shap_values = explainer.shap_values(test_samples)
    
    # Calculate mean absolute SHAP values for both classes
    shap_importance = np.abs(shap_values[1]).mean(axis=0)
    
    # # Create DataFrame with importance metrics
    # importance_df = pd.DataFrame({
    #     'Feature': feature_names,
    #     'SHAP_Importance': shap_importance
    # })
    
    importance_df = pd.DataFrame({
    'Feature': feature_names[:len(shap_importance)],  # Ensure alignment
    'SHAP_Importance': shap_importance
    })

    # Sort by SHAP importance
    importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
    
    # Save to Excel
    importance_df.to_excel('feature_importance_mlp.xlsx', index=False)
    
    # Create bar plot of SHAP importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_names)), importance_df['SHAP_Importance'])
    plt.xticks(range(len(feature_names)), importance_df['Feature'], rotation=45, ha='right')
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('shap_importance_bar_mlp.png')
    plt.close()
    
    # Top 20 Feature Dependencies Plot
    top_features = importance_df['Feature'].head(5).tolist()
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        feature_index = list(feature_names).index(feature)
        shap.dependence_plot(feature_index, shap_values[1], test_samples, 
                           feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot for {feature}')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature.replace("/", "_")}.png')
        plt.close()
    
    # Print top features
    print("\nTop 10 Features by SHAP Importance:")
    print(importance_df.head(10))
    
    return importance_df, shap_values

# Perform feature importance analysis
print("\nPerforming feature importance and SHAP analysis...")
importance_df, shap_values = analyze_feature_importance_and_shap(
    model=mlp,
    X_train_df=X_train_scaled_df,
    X_test_df=X_test_scaled_df,
    feature_names=features
)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_mlp.png')
plt.close()

# Feature correlations
plt.figure(figsize=(125, 125))
correlation_matrix = np.corrcoef(X_test_scaled.T)
sns.heatmap(
    correlation_matrix,
    xticklabels=features,
    yticklabels=features,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt='.2f'
)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlations_mlp.png')
plt.close()

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(mlp.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curve_mlp.png')
plt.close()

print("\nAnalysis complete! Files saved:")
print("1. shap_importance_bar_mlp.png - SHAP feature importance bar plot")
print("2. feature_importance_mlp.xlsx - Detailed feature importance scores")
print("3. feature_correlations_mlp.png - Feature correlation matrix")
print("4. roc_curve_mlp.png - ROC curve")
print("5. learning_curve_mlp.png - Learning curve")
print("6. shap_dependence_*.png - SHAP dependence plots for top features")

# Print model performance metrics
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc:.3f}")
print(f"Number of iterations to converge: {mlp.n_iter_}")

# Save feature importance to text file
with open('feature_importance_summary.txt', 'w') as f:
    f.write("Feature Importance Summary\n")
    f.write("=========================\n\n")
    f.write("Top 10 Most Important Features:\n")
    for idx, row in importance_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['SHAP_Importance']:.4f}\n")

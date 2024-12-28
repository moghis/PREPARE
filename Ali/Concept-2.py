# Classification Task with Concept Autoencoder
# Target: CDRGLOB
# Run correct. Problem with results
# SCAN dataset 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
############################################################################################################
def evaluate_model(model, X_tensor, y_true):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        predictions = (torch.sigmoid(logits) >= 0.5).float().numpy()
    
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary')
    conf_matrix = confusion_matrix(y_true, predictions)
    
    return accuracy, precision, recall, f1, conf_matrix

############################################################################################################
class ConceptAutoencoder(nn.Module):
    def __init__(self, concept_dims, latent_dim=4, dropout_rate=0.5):
        super(ConceptAutoencoder, self).__init__()
        self.concept_dims = concept_dims
        self.latent_dim = latent_dim
        
        self.concept_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, max(dim, latent_dim)),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(max(dim, latent_dim), latent_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for dim in concept_dims
        ])
        
        total_input_dim = sum(concept_dims)
        self.decoder = nn.Sequential(
            nn.Linear(len(concept_dims) * latent_dim, total_input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(total_input_dim // 2, total_input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        concept_inputs = torch.split(x, self.concept_dims, dim=1)
        encoded_concepts = [encoder(concept_input) for encoder, concept_input in zip(self.concept_encoders, concept_inputs)]
        latent = torch.cat(encoded_concepts, dim=1)
        decoded = self.decoder(latent)
        return decoded, latent
   
############################################################################################################
class ConceptMLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ConceptMLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(30, 1)  # Single output for regression
        )
    
    def forward(self, x):
        return self.model(x)
############################################################################################################

def load_and_preprocess_data(file_path, features, target, concept_groups):
    data = pd.read_excel(file_path)
    data[features] = data[features].apply(pd.to_numeric, errors='coerce')
    data[features] = data[features].fillna(data[features].median())
    
    # Target as a continuous variable
    data[target] = pd.to_numeric(data[target], errors='coerce')
    
    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    concept_indices = [features.index(feature) for group in concept_groups for feature in group]
    X_scaled_reordered = X_scaled[:, concept_indices]
    
    return X_scaled_reordered, y, scaler

############################################################################################################
def train_concept_autoencoder(autoencoder, X_train_tensor, num_epochs=200, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        for data in train_loader:
            inputs, _ = data
            outputs, _ = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return autoencoder
############################################################################################################

def evaluate_regression_model(model, X_tensor, y_true):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }
############################################################################################################
def train_concept_regressor(regressor, X_train_tensor, y_train_tensor, num_epochs=100, batch_size=32):
    criterion = nn.MSELoss()  # Regression loss
    optimizer = optim.Adam(regressor.parameters(), lr=0.001, weight_decay=1e-5)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        regressor.train()
        total_loss = 0
        for data in train_loader:
            inputs, targets = data
            outputs = regressor(inputs).view(-1)
            loss = criterion(outputs, targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return regressor
############################################################################################################

if __name__ == "__main__":
    
    datasets = ['/home/data3/Ali/Code/Moghis/Train-data-preprocess-2.xlsx']
    datasets_test = ['/home/data3/Ali/Code/Moghis/Test-data-preprocess-2.xlsx']
    
    features_list = [
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

    concept_groups_list = [
    # Demographic Information
    [["age_03", "age_12"], ["urban_03", "urban_12"], ["married_03", "married_12"],
     ["n_mar_03", "n_mar_12"], ["ragender"], ["sgender_03", "sgender_12"]],

    # Socioeconomic Status
    [["edu_gru_03", "edu_gru_12"], ["employment_03", "employment_12"],
     ["rearnings_03", "rearnings_12"], ["searnings_03", "searnings_12"],
     ["hincome_03", "hincome_12"], ["hinc_business_03", "hinc_business_12"],
     ["hinc_rent_03", "hinc_rent_12"], ["hinc_assets_03", "hinc_assets_12"],
     ["hinc_cap_03", "hinc_cap_12"], ["rinc_pension_03", "rinc_pension_12"],
     ["sinc_pension_03", "sinc_pension_12"]],

    # Health and Physical Limitations
    [["glob_hlth_03", "glob_hlth_12"], ["adl_dress_03", "adl_dress_12"],
     ["adl_walk_03", "adl_walk_12"], ["adl_bath_03", "adl_bath_12"],
     ["adl_eat_03", "adl_eat_12"], ["adl_bed_03", "adl_bed_12"],
     ["adl_toilet_03", "adl_toilet_12"], ["n_adl_03", "n_adl_12"],
     ["iadl_money_03", "iadl_money_12"], ["iadl_meds_03", "iadl_meds_12"],
     ["iadl_shop_03", "iadl_shop_12"], ["iadl_meals_03", "iadl_meals_12"],
     ["n_iadl_03", "n_iadl_12"], ["hypertension_03", "hypertension_12"],
     ["diabetes_03", "diabetes_12"], ["resp_ill_03", "resp_ill_12"],
     ["arthritis_03", "arthritis_12"], ["hrt_attack_03", "hrt_attack_12"],
     ["stroke_03", "stroke_12"], ["cancer_03", "cancer_12"],
     ["n_illnesses_03", "n_illnesses_12"], ["bmi_12"]],

    # Mental Health and Well-being
    [["depressed_03", "depressed_12"], ["hard_03", "hard_12"],
     ["restless_03", "restless_12"], ["happy_03", "happy_12"],
     ["lonely_03", "lonely_12"], ["enjoy_03", "enjoy_12"],
     ["sad_03", "sad_12"], ["tired_03", "tired_12"],
     ["energetic_03", "energetic_12"], ["n_depr_03", "n_depr_12"],
     ["cesd_depressed_03", "cesd_depressed_12"]],

    # Lifestyle and Behavior
    [["exer_3xwk_03", "exer_3xwk_12"], ["alcohol_03", "alcohol_12"],
     ["tobacco_03", "tobacco_12"], ["test_chol_03", "test_chol_12"],
     ["test_tuber_03", "test_tuber_12"], ["test_diab_03", "test_diab_12"],
     ["test_pres_03", "test_pres_12"], ["hosp_03", "hosp_12"],
     ["visit_med_03", "visit_med_12"], ["out_proc_03", "out_proc_12"],
     ["visit_dental_03", "visit_dental_12"]],

    # Social and Family Dynamics
    [["n_living_child_03", "n_living_child_12"], ["migration_03", "migration_12"],
     ["decis_famil_12"], ["decis_personal_03", "decis_personal_12"],
     ["care_adult_12"], ["care_child_12"], ["volunteer_12"], ["attends_class_12"],
     ["attends_club_12"], ["reads_12"], ["games_12"], ["table_games_12"],
     ["comms_tel_comp_12"], ["act_mant_12"], ["tv_12"], ["sewing_12"],
     ["satis_ideal_12"], ["satis_excel_12"], ["satis_fine_12"],
     ["cosas_imp_12"], ["wouldnt_change_12"], ["memory_12"], 
     ["rrelgimp_03", "rrelgimp_12"], ["rrfcntx_m_12"], ["rsocact_m_12"], ["rrelgwk_12"]],

    # Health Insurance and Coverage
    [["imss_03", "imss_12"], ["issste_03", "issste_12"],
     ["pem_def_mar_03", "pem_def_mar_12"], ["insur_private_03", "insur_private_12"],
     ["insur_other_03", "insur_other_12"], ["seg_pop_12"], ["insured_03", "insured_12"]],

    # Migration and U.S. Experience
    [["a34_12"]],

    # Housing and Environment
    [["j11_12"]],

    # Parental Education
    [["rameduc_m"], ["rafeduc_m"]],

    # Composite Scores and Changes
    [["composite_score"], ["hincome_change"], ["niadl_change"],
     ["adl_change"], ["depr_change"], ["glob_hlth_change"],
     ["edu_gru_change"], ["illnesses_change"]],

    # Identifier
    # [["uid"], ["year"]]
    [["year"]]
    ]


    target = 'composite_score'
    
    # Load and preprocess data
    X_scaled, y, scaler = load_and_preprocess_data(datasets[0], features_list, target, concept_groups_list[0])  # Load Train Data
    X_scaled_test, y_test, scaler_test = load_and_preprocess_data(datasets_test[0], features_list, target, concept_groups_list[0])  # Load Test Data

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    
    X_test_tensor = torch.tensor(X_scaled_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Train autoencoder
    concept_dims = [len(group) for group in concept_groups_list[0]]
    autoencoder = ConceptAutoencoder(concept_dims)
    autoencoder = train_concept_autoencoder(autoencoder, X_train_tensor)
    
    # Get encoded representations and train classifier
    with torch.no_grad():
        _, encoded_X_train = autoencoder(X_train_tensor)
    
    regressor = ConceptMLPRegressor(input_dim=encoded_X_train.shape[1])
    regressor = train_concept_regressor(regressor, encoded_X_train, y_train_tensor)

############################################################################################################



    with torch.no_grad():
        _, encoded_X_test = autoencoder(X_test_tensor)
    test_metrics = evaluate_regression_model(regressor, encoded_X_test, y_test.values)

    print("\nTest Set Metrics:")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R^2: {test_metrics['r2']:.4f}")

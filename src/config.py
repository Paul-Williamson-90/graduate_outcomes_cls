from enum import Enum

class Config(Enum):
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    target = "Target"
    categoricals = [
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International"
    ]
    numericals = [
        "Previous qualification (grade)",
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Unemployment rate",
        "Inflation rate",
        "GDP"
    ]
    val_split = 0.2
    hidden_units = [128, 64, 32]
    learning_rate = 1e-4
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    eval_steps = 0.2
    early_stopping_patience = 5
    batch_size = 32
rec_data_path = "data/recommender_data.csv"
pref_med_path = "data/prefecture_medians.csv"
company_path = "data/company_list_final.csv"

grade_values_mapping = {
    'Futsū-shu': 0,
    'Honjōzō-shu': 1,
    'Junmai-shu': 1,
    'Junmai-kei': 1,
    'Honjōzō-kei': 1,
    'Tokubetsu Honjōzō-shu': 2,
    'Tokubetsu Junmai-shu': 2,
    'Ginjō-shu': 3,
    'Junmai Ginjō-shu': 3,
    'Daiginjō-shu': 4,
    'Junmai Daiginjō-shu': 4
}

junmai_mapping = {
    'Futsū-shu': 0,
    'Honjōzō-shu': 0,
    'Junmai-shu': 1,
    'Junmai-kei': 1,
    'Honjōzō-kei': 0,
    'Tokubetsu Honjōzō-shu': 0,
    'Tokubetsu Junmai-shu': 1,
    'Ginjō-shu': 0,
    'Junmai Ginjō-shu': 1,
    'Daiginjō-shu': 0,
    'Junmai Daiginjō-shu': 1
}

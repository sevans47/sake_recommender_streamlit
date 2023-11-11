import pandas as pd
import numpy as np
import streamlit as st
import folium
from typing import Tuple
from streamlit_folium import folium_static
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from const import rec_data_path, pref_med_path, company_path, grade_values_mapping, junmai_mapping

# st.cache_data.clear()

st.set_page_config(
    page_title="Sake Recommender",
    page_icon='🍶',
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)


@st.cache_data
def fetch_data(
    rec_path: str,
    pref_medians_path: str,
    company_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rec_df = pd.read_csv(rec_path)
    pref_medians_df = pd.read_csv(pref_medians_path, index_col="prefecture").drop_duplicates()
    company_df = pd.read_csv(company_path)
    return (rec_df.dropna(), pref_medians_df, company_df.dropna(subset="address"))


@st.cache_data
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    df = df[['abv_avg', 'acidity_avg', 'gravity_avg', 'rice_polishing_rate', 'grade', 'is_junmai', 'lat', 'lon', 'dist_from_coast_km', 'elevation']]
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return (df, scaler)


def get_rec(
    abv, acidity, smv, polish_rate, grade, is_junmai, lat, lon, dist_from_coast, elevation, scaler, preproc_df, raw_df
) -> pd.DataFrame:
    user_sake = np.array([[
        abv, acidity, smv, polish_rate, grade, is_junmai, lat, lon, dist_from_coast, elevation
    ]])
    user_sake = scaler.transform(user_sake)
    df = np.vstack([preproc_df, user_sake])
    top_n = 5
    similarities = cosine_similarity(df)[-1][:-1]
    top_indices = np.argpartition(similarities, -top_n)[-top_n:]
    top_indices = top_indices[::-1]
    top_similarities = similarities[top_indices]
    top_sake = raw_df.iloc[top_indices]
    top_similarities = pd.Series(top_similarities, top_sake.index)
    top_sake = pd.concat((top_sake, top_similarities), axis=1)
    top_sake = top_sake.rename(columns={
        'name_romaji': "sake_name", "prefecture_eng": "prefecture",
        "abv_avg": "abv", "acidity_avg": "acidity", "gravity_avg": "smv",
        "dist_from_coast_km": "dist_from_coast", 0: "similarity_score"
    })
    top_sake = top_sake.reindex(columns=[
        "similarity_score", "sake_name", "company", "prefecture", "abv",
        "acidity", "smv", "rice_polishing_rate", "is_junmai", "grade",
        "lat", "lon", "dist_from_coast", "elevation"
    ])

    st.session_state['top_sake'] = top_sake

    # return top_sake


raw_df, pref_medians_df, company_df = fetch_data(rec_data_path, pref_med_path, company_path)
preproc_df, scaler = prepare_data(raw_df)


# map variables
tile_light_gray = 'https://server.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
attr_light_gray = 'Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community'
test_coord = [39, 140]

def make_map():
    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )
    gj_data = "data/jp_prefs.geojson"
    gj = folium.GeoJson(gj_data)
    gj.add_to(m)

    def plotDot(point):
        folium.Marker(
            location=[point.lat, point.lon],
            radius=10,
            weight=5,
            popup=f"Sake name:\n {point.sake_name}\n Company:\n {point.company}\n Prefectures:\n {point.prefecture}"
        ).add_to(m)

    st.session_state['top_sake'].apply(plotDot, axis=1)

    return m


if 'top_sake' not in st.session_state:
    st.session_state['top_sake'] = None
if 'company_search' not in st.session_state:
    st.session_state['company_search'] = None
if 'pref_search' not in st.session_state:
    st.session_state['pref_search'] = None
if 'val_cols' not in st.session_state:
    st.session_state['val_cols'] = [
        "abv_med", "acidity_med", "gravity_med", "polish_rate_med", "grade_med", "lat", "lon",
        "elevation", "dist_from_coast_km"
    ]



# page layout

st.markdown("# Sake Recommender :sake:")
st.subheader("Input information for a bottle of sake to be recommended similar bottles of sake")

search_type = st.selectbox(
"Input sake info by brewery or by prefecture?",
("By Brewery", "By Prefecture"),
index=None,
placeholder="Select one",
)

if search_type == "By Brewery":
    st.session_state['pref_search'] = None
    # companies = company_df["company"].tolist()
    companies = [f"{jp} / {eng}" for jp, eng in zip(company_df["company"], company_df["company_eng"])]
    st.session_state['company_search'] = st.selectbox(
        "Which company is the sake made by?",
        companies,
        index=None,
        placeholder="Start typing company name in Japanese"
    )
    if st.session_state['company_search']:
        chosen_company = company_df[company_df["company"]==st.session_state['company_search'].split(" / ")[0]]

        # use pref medians for any nan values in chosen_company
        company_pref = pref_medians_df.loc[chosen_company["prefecture_eng"].values[0]]
        company_vals = pd.Series(chosen_company[st.session_state['val_cols']].values[0], index=chosen_company[st.session_state['val_cols']].columns)
        company_pref_vals = pref_medians_df[st.session_state['val_cols']].loc[chosen_company["prefecture_eng"].values[0]]
        default_vals = company_vals.fillna(company_pref_vals)

        st.write(f"Company Info for {st.session_state['company_search']}:")
        st.dataframe(chosen_company.set_index("company")[["prefecture_eng", "region", "address", "website", "num_sake"]])

elif search_type == "By Prefecture":
    st.session_state['company_search'] = None
    prefectures = pref_medians_df.index.tolist()
    st.session_state['pref_search'] = st.selectbox(
        "Which prefecture is the sake made in?",
        prefectures,
        index=None,
        placeholder="Start typing prefecture name in English"
    )
    if st.session_state['pref_search']:
        default_vals = pref_medians_df[st.session_state['val_cols']].loc[st.session_state['pref_search']]


# TODO:
# show st.number_input() only once the pref / brewery is selected - DONE
# show default values based on pref / brewery info - DONE
# set location data without showing it - DONE
# add median values for polish_rate, grade, and junmai for company and prefecture csvs - DONE
# allow for English search for brewery?  Add English column to company_list_final.csv, each company in selectbox is "日本語名前 / English name"


if any((st.session_state['company_search'] is not None, st.session_state['pref_search'] is not None)):
    if st.session_state['company_search'] is not None:
        location = st.session_state['company_search']
    elif st.session_state['pref_search'] is not None:
        location = st.session_state['pref_search']
    st.subheader("Please input additional information below")
    st.write(f"If you're unsure, feel free to use the median values for {location}, which are in parentheses.")
    abv = st.number_input(f"Alcohol % ({default_vals['abv_med']:.2f}):", value=default_vals["abv_med"])
    acidity = st.number_input(f"Acidity ({default_vals['acidity_med']:.2f}):", value=default_vals["acidity_med"])
    smv = st.number_input(f"Sake Meter Value ({default_vals['gravity_med']:.2f}):", value=default_vals["gravity_med"])
    polish_rate = st.number_input(f"Rice Polishing  ({default_vals['polish_rate_med']:.2f}):", value=default_vals['polish_rate_med'])

    grades = list(grade_values_mapping.keys())
    grade_idx = grades.index(default_vals['grade_med'])
    grade_name = st.selectbox(f"Grade ({default_vals['grade_med']}):", grades, index=grade_idx)

    grade = grade_values_mapping[grade_name]
    is_junmai = junmai_mapping[grade_name]
    lat = default_vals["lat"]
    lon = default_vals["lon"]
    dist_from_coast = default_vals["dist_from_coast_km"]
    elevation = default_vals["elevation"]


    if st.button('Get Recommendation!', on_click=get_rec, args=(
        abv, acidity, smv, polish_rate, grade, is_junmai, lat, lon, dist_from_coast, elevation, scaler, preproc_df, raw_df
    )):
        st.dataframe(st.session_state['top_sake'])
        m = make_map()
        folium_static(m)
        st.markdown('[Back to Top](#sake-recommender)')
        # if st.button("Click to start over!"):
        #     st.session_state['top_sake'] = None
        #     st.session_state['pref_search'] = None
        #     st.session_state['company_search'] = None
        #     placeholder.empty()



if __name__ == "__main__":
    companies = company_df["company"].tolist()

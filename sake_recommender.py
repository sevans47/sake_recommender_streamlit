import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import folium_static
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Sake Recommender",
    page_icon='🍶',
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)


@st.cache_data
def fetch_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna()


@st.cache_data
def prepare_data(df: pd.DataFrame) -> tuple:
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


data_path = "data/recommender_data.csv"
raw_df = fetch_data(data_path)
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


# page layout
st.title("Sake Recommender")
st.subheader("Input Sake Information:")


abv = st.number_input("Alcohol %:", value=raw_df["abv_avg"].median())
acidity = st.number_input("Acidity:", value=raw_df["acidity_avg"].median())
smv = st.number_input("Sake Meter Value:", value=raw_df["gravity_avg"].median())
polish_rate = st.number_input("Rice Polishing Rate:", value=raw_df["rice_polishing_rate"].median())
grade = st.number_input("Grade:", min_value=0, max_value=4, value=int(raw_df["grade"].median()), step=1)
is_junmai = st.number_input("Is Junmai:", min_value=0, max_value=1, value=int(raw_df["is_junmai"].median()), step=1)
lat = st.number_input("Latitude:", value=raw_df["lat"].median())
lon = st.number_input("Longitude", value=raw_df["lon"].median())
dist_from_coast = st.number_input("Distance from Coast (km):", value=raw_df["dist_from_coast_km"].median())
elevation = st.number_input("Elevation (m):", value=raw_df["elevation"].median())


if st.button('Get Recommendation!', on_click=get_rec, args=(
    abv, acidity, smv, polish_rate, grade, is_junmai, lat, lon, dist_from_coast, elevation, scaler, preproc_df, raw_df
)):
    st.dataframe(st.session_state['top_sake'])
    m = make_map()
    folium_static(m)

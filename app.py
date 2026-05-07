import streamlit as st
import pandas as pd
import pickle
import folium
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Brisbane CBD Smart Stop Advisor", page_icon="🅿️", layout="wide")

@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('zone_avg.pkl', 'rb') as f:
        zone_avg = pickle.load(f)
    meters  = pd.read_csv('raw/brisbane-parking-meters.csv')
    bus     = pd.read_csv('raw/brisbane-bus-stops.csv')
    loading = pd.read_csv('raw/two-minute-passenger-loading-zones.csv')
    return model, zone_avg, meters, bus, loading

model, zone_avg, meters_raw, bus_raw, loading_raw = load_assets()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

@st.cache_data
def prepare_data():
    CBD_LAT, CBD_LON = -27.4698, 153.0251
    def filt(df, la, lo):
        df = df.dropna(subset=[la, lo]).copy()
        df['d'] = df.apply(lambda r: haversine(CBD_LAT, CBD_LON, r[la], r[lo]), axis=1)
        return df[df['d'] <= 3500].reset_index(drop=True)
    m = filt(meters_raw, 'LATITUDE', 'LONGITUDE')
    b = filt(bus_raw, 'LATITUDE', 'LONGITUDE')
    l = filt(loading_raw, 'Latitude', 'Longitude')
    l = l[(l['Parking_Restriction_Type']=='Loading Zone Passengers 2 Min. Max') & (l['Sign Status']=='In Service')]
    return m, b, l

meters_cbd, bus_cbd, loading_cbd = prepare_data()

def predict_availability(mobile_zone, query_time):
    hour, dow = query_time.hour, query_time.weekday()
    match = zone_avg[(zone_avg['MOBILE_ZONE']==mobile_zone) & (zone_avg['HOUR']==hour) & (zone_avg['day_of_week']==dow)]
    zav = match['zone_hour_avg'].values[0] if len(match) > 0 else 2.5
    features = pd.DataFrame([[hour, dow, int(dow<5), int(7<=hour<=9),
                               int(16<=hour<=18), int(9<=hour<=17), zav]],
                             columns=['HOUR','day_of_week','is_weekday',
                                      'is_peak_am','is_peak_pm','is_business','zone_hour_avg'])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return pred, proba

def is_loading_legal(text, query_time):
    if not isinstance(text, str) or text.strip() == '':
        return True
    hour, dow = query_time.hour, query_time.weekday()
    if 'SCHOOL' in text.upper():
        if dow < 5 and (7 <= hour <= 9 or 14 <= hour <= 16):
            return False
    return True

def recommend(dest_lat, dest_lon, query_time, radius_m=400):
    results = []
    labels = ['Likely available', '50/50', 'Avoid']
    colors = ['green', 'orange', 'red']

    for _, row in meters_cbd.iterrows():
        dist = haversine(dest_lat, dest_lon, row['LATITUDE'], row['LONGITUDE'])
        if dist > radius_m:
            continue
        pred, proba = predict_availability(row['MOBILE_ZONE'], query_time)
        results.append({'type':'Paid Parking', 'street':row['STREET'],
            'distance_m':round(dist), 'max_stay':f"{row['MAX_STAY_HRS']}hr max",
            'prediction':labels[pred], 'confidence':round(max(proba)*100),
            'tier':pred, 'lat':row['LATITUDE'], 'lon':row['LONGITUDE'], 'color':colors[pred]})

    for _, row in loading_cbd.iterrows():
        dist = haversine(dest_lat, dest_lon, row['Latitude'], row['Longitude'])
        if dist > radius_m:
            continue
        legal = is_loading_legal(row['ParkingRestrictionDaysandTimes'], query_time)
        results.append({'type':'2-Min Loading Zone', 'street':row['Street'],
            'distance_m':round(dist), 'max_stay':'2 min max',
            'prediction':'Legal now' if legal else 'Restricted now',
            'confidence':100, 'tier':0 if legal else 2,
            'lat':row['Latitude'], 'lon':row['Longitude'],
            'color':'blue' if legal else 'red'})

    for _, row in bus_cbd.iterrows():
        dist = haversine(dest_lat, dest_lon, row['LATITUDE'], row['LONGITUDE'])
        if dist > radius_m:
            continue
        results.append({'type':'Bus Stop', 'street':row['STREETNAME'],
            'distance_m':round(dist), 'max_stay':'Never legal',
            'prediction':'Do not stop', 'confidence':100, 'tier':2,
            'lat':row['LATITUDE'], 'lon':row['LONGITUDE'], 'color':'red'})

    results.sort(key=lambda x: (x['tier'], x['distance_m']))
    return results

# UI
st.title("🅿️ Brisbane CBD Smart Stop Advisor")
st.caption("Find the best legal parking or stopping option — based on time of day and historical patterns.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your destination")
    destinations = {
        "Queen Street Mall"   : (-27.4698, 153.0278),
        "South Bank"          : (-27.4820, 153.0217),
        "Roma Street Station" : (-27.4655, 153.0189),
        "Brisbane City Hall"  : (-27.4679, 153.0235),
        "Central Station"     : (-27.4641, 153.0262),
    }
    dest_choice = st.selectbox("Select destination", list(destinations.keys()))
    dest_lat, dest_lon = destinations[dest_choice]

    time_mode = st.radio("When are you arriving?", ["Right now", "Choose a time"], horizontal=True)

    if time_mode == "Right now":
        query_time = datetime.now()
        st.info(f"Current time: {query_time.strftime('%I:%M %p, %A')}")
    else:
        d = st.date_input("Date", value=datetime.now().date())
        hour = st.selectbox("Hour", list(range(6, 23)), index=3)
        minute = st.selectbox("Minute", [0, 15, 30, 45], index=0)
        query_time = datetime.combine(d, datetime.min.time().replace(hour=hour, minute=minute))
        st.info(f"Selected: {query_time.strftime('%I:%M %p, %A')}")

    radius = st.slider("Search radius (metres)", 200, 800, 400, step=50)
    search = st.button("🔍 Find parking options", use_container_width=True)

with col2:
    if search:
        with st.spinner("Analysing parking options..."):
            results = recommend(dest_lat, dest_lon, query_time, radius)

        best     = [r for r in results if r['tier'] == 0]
        possible = [r for r in results if r['tier'] == 1]
        avoid    = [r for r in results if r['tier'] == 2]

        st.subheader(f"Results — {dest_choice} at {query_time.strftime('%I:%M %p, %A')}")
        st.caption(f"{len(results)} options within {radius}m · Historical patterns, not live data")

        m = folium.Map(location=[dest_lat, dest_lon], zoom_start=16)
        folium.Marker([dest_lat, dest_lon], popup="Destination",
                      icon=folium.Icon(color='purple', icon='star')).add_to(m)
        for r in results:
            folium.CircleMarker(
                location=[r['lat'], r['lon']], radius=7,
                color=r['color'], fill=True, fill_opacity=0.7,
                popup=f"{r['type']} - {r['street']} - {r['prediction']} - {r['distance_m']}m"
            ).add_to(m)
        st.components.v1.html(m._repr_html_(), height=400)

        if best:
            st.success(f"✅ Best options ({len(best)})")
            for r in best[:6]:
                st.markdown(f"**{r['type']}** · {r['street']} · {r['distance_m']}m · {r['prediction']} ({r['confidence']}%) · {r['max_stay']}")

        if possible:
            st.warning(f"🟡 Possible options ({len(possible)})")
            for r in possible[:5]:
                st.markdown(f"**{r['type']}** · {r['street']} · {r['distance_m']}m · {r['prediction']} ({r['confidence']}%) · {r['max_stay']}")

        if avoid:
            with st.expander(f"🔴 Avoid ({len(avoid)} options)"):
                for r in avoid[:8]:
                    st.markdown(f"**{r['type']}** · {r['street']} · {r['distance_m']}m · {r['prediction']}")

        st.caption("⚠️ Always check local signs. Restrictions can change.")
    else:
        st.info("👈 Select a destination and click 'Find parking options'")
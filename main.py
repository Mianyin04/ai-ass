import streamlit as st
import joblib
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Navigation")
model_option = st.sidebar.selectbox("Select Recommendation Approach", [
    "Approach 1: Random Forest",
    "Approach 2: KNN + Linear Regression",
    "Approach 3: Naive Bayes + Decision Tree"
])

laptop_data = pd.read_csv("laptops_updated.csv")
laptop_data2 = pd.read_csv("laptops.csv")

# Load corresponding model file
if model_option == "Approach 1: Random Forest":
    model_data = joblib.load("approach1_rf.joblib")
elif model_option == "Approach 2: KNN + Linear Regression":
    model_data = joblib.load("approach2_knn_lr.joblib")
else:
    model_data = joblib.load("approach3_nb_dt.joblib")

# Extract components
clf = model_data["classification_model"]
reg = model_data["regression_model"]
label_encoders = model_data["label_encoders"]
features = model_data["features"]

# Define possible values
# possible_display_sizes = ["13.3", "14.0", "15.6", "16.0", "17.3"]
# possible_resolution_widths = ["1366", "1920", "2560", "3840"]
# possible_resolution_heights = ["768", "1080", "1440", "2160"]
# possible_num_cores = list(range(2, 17, 2))
# possible_num_threads = list(range(2, 17, 2))
# possible_ram_memory = list(range(4, 33, 4))

# Sidebar Inputs
st.sidebar.title("Laptop Recommendation System")
st.sidebar.write("Enter your preferences and click below!")

user_input = {}

# Select Brand First
brand_options = sorted(laptop_data2["brand"].unique().tolist())
selected_brand = st.sidebar.selectbox("Select Brand", brand_options)

# Filter processor brands dynamically
filtered_processor_brands = sorted(laptop_data2[laptop_data2["brand"] == selected_brand]["processor_brand"].unique().tolist())
selected_processor_brand = st.sidebar.selectbox("Select Processor Brand", filtered_processor_brands)

filtered_processor_tiers = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand)
    ]["processor_tier"].unique().tolist(),
    key=lambda x: [int(s) if s.isdigit() else s for s in x.split()]
)
selected_processor_tier = st.sidebar.selectbox("Select Processor Tier", filtered_processor_tiers)

# Filter Number of Cores Based on Processor Tier Selection
filtered_num_cores = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand) & 
        (laptop_data2["processor_tier"] == selected_processor_tier)
    ]["num_cores"].unique().tolist()
)
selected_num_cores = st.sidebar.selectbox("Select Number of Cores", filtered_num_cores)

# Filter Number of Threads Based on Cores Selection
filtered_num_threads = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand) & 
        (laptop_data2["processor_tier"] == selected_processor_tier) & 
        (laptop_data2["num_cores"] == selected_num_cores)
    ]["num_threads"].unique().tolist()
)
selected_num_threads = st.sidebar.selectbox("Select Number of Threads", filtered_num_threads)

# Filter RAM Memory Based on Processor Selection
filtered_ram_memory = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand) & 
        (laptop_data2["processor_tier"] == selected_processor_tier) &
        (laptop_data2["num_cores"] == selected_num_cores) & 
        (laptop_data2["num_threads"] == selected_num_threads)
    ]["ram_memory"].unique().tolist()
)
selected_ram_memory = st.sidebar.selectbox("Select RAM Memory", filtered_ram_memory)

# Filter GPU Brand Based on Brand Selection
filtered_gpu_brands = sorted(laptop_data2[laptop_data2["brand"] == selected_brand]["gpu_brand"].unique().tolist())
selected_gpu_brand = st.sidebar.selectbox("Select GPU Brand", filtered_gpu_brands)

# Filter GPU Type Based on GPU Brand Selection
filtered_gpu_brands = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand) & 
        (laptop_data2["processor_tier"] == selected_processor_tier) & 
        (laptop_data2["num_cores"] == selected_num_cores) & 
        (laptop_data2["num_threads"] == selected_num_threads) & 
        (laptop_data2["ram_memory"] == selected_ram_memory)
    ]["gpu_brand"].unique().tolist()
)
selected_gpu_type = st.sidebar.selectbox("Select GPU Type", filtered_gpu_types)

# Filter Display Size Based on Brand Selection
filtered_display_sizes = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand) & 
        (laptop_data2["processor_tier"] == selected_processor_tier) & 
        (laptop_data2["num_cores"] == selected_num_cores) & 
        (laptop_data2["num_threads"] == selected_num_threads) & 
        (laptop_data2["ram_memory"] == selected_ram_memory) & 
        (laptop_data2["gpu_brand"] == selected_gpu_brand) & 
        (laptop_data2["gpu_type"] == selected_gpu_type)
    ]["display_size"].unique().tolist(),
    reverse=True
)
selected_display_size = st.sidebar.selectbox("Select Display Size", filtered_display_sizes)

# Filter Resolution Width Based on Display Size Selection
filtered_resolution_widths = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["display_size"] == selected_display_size)
    ]["resolution_width"].unique().tolist()
)
selected_resolution_width = st.sidebar.selectbox("Select Resolution Width", filtered_resolution_widths)

# Filter Resolution Height Based on Width Selection
filtered_resolution_heights = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["display_size"] == selected_display_size) & 
        (laptop_data2["resolution_width"] == selected_resolution_width)
    ]["resolution_height"].unique().tolist()
)
selected_resolution_height = st.sidebar.selectbox("Select Resolution Height", filtered_resolution_heights)

# Filter OS Based on Brand Selection
filtered_os = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["display_size"] == selected_display_size) & 
        (laptop_data2["resolution_width"] == selected_resolution_width) & 
        (laptop_data2["resolution_height"] == selected_resolution_height)
    ]["OS"].unique().tolist()
)
selected_os = st.sidebar.selectbox("Select Operating System", filtered_os)

# Store user input
user_input.update({
    "brand": selected_brand,
    "processor_brand": selected_processor_brand,
    "processor_tier": selected_processor_tier,
    "num_cores": selected_num_cores,
    "num_threads": selected_num_threads,
    "ram_memory": selected_ram_memory,
    "gpu_brand": selected_gpu_brand,
    "gpu_type": selected_gpu_type,
    "display_size": selected_display_size,
    "resolution_width": selected_resolution_width,
    "resolution_height": selected_resolution_height,
    "OS": selected_os
})

predict_button = st.sidebar.button("Get Recommendation")

# Main Output
st.title("Laptop Recommendation System")
st.write("Recommendations will appear below based on your selected preferences.")

if predict_button:
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=features, fill_value=0)

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df = input_df.astype(float)

    try:
        classification_prediction = int(clf.predict(input_df)[0])
        regression_prediction = reg.predict(input_df)[0]

        category_labels = {1: "Gaming", 2: "Business", 3: "Budget-Friendly"}
        category_name = category_labels.get(classification_prediction, "Unknown")

        st.subheader("Recommendation Results:")
        st.write(f"Predicted Laptop Category: {category_name}")
        st.write(f"Estimated Laptop Price: ${regression_prediction:.2f}")

        matching_laptops = laptop_data[laptop_data['Category'] == classification_prediction]
        if not matching_laptops.empty:
            st.subheader("Possible Laptop Models:")
            st.write(matching_laptops[['Model', 'brand', 'Price']])
        else:
            st.write("No matching laptops found in the dataset.")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

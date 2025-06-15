import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="ICMR Data Visualization", layout="wide")

# Title and description
st.title("ICMR Maternal and Newborn Data Dashboard")
st.markdown("Upload your CSV to compare Haemoglobin, Blood Group, Cord Blood, Head Circumference, Length, and Weight.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Define expected column names
    mother_haemoglobin_col = 'Haemoglobin of Mother '
    blood_group_col = 'Blood Group of Mother '
    cord_haemoglobin_col = 'Cord Blood Haemoglobin '
    head_circum_day1_col = 'Head Circumference of Baby on Day 1(In cm)'
    head_circum_day3_col = 'Head Circumference of Baby on Day 3(In cm)'
    length_col = 'Length of Baby(in cm)'
    weight_col = 'Weight of Baby(in kg)'
    sex_col = 'Sex Of Baby'
    address_col = 'Address of Mother'
    parity_col = 'Parity of Mother'

    # Check if required columns exist
    required_cols = [mother_haemoglobin_col, blood_group_col, cord_haemoglobin_col, head_circum_day1_col, head_circum_day3_col, length_col, weight_col, sex_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # Standardize categorical columns
    df[sex_col] = df[sex_col].str.lower().str.strip()
    df[blood_group_col] = df[blood_group_col].str.strip().str.upper()

    # Clean numerical data
    df[mother_haemoglobin_col] = pd.to_numeric(df[mother_haemoglobin_col], errors='coerce')
    df[cord_haemoglobin_col] = pd.to_numeric(df[cord_haemoglobin_col], errors='coerce')
    df[head_circum_day1_col] = pd.to_numeric(df[head_circum_day1_col], errors='coerce')
    df[head_circum_day3_col] = pd.to_numeric(df[head_circum_day3_col], errors='coerce')
    df[length_col] = pd.to_numeric(df[length_col], errors='coerce')
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filter Options")
    mother_haemoglobin_range = st.sidebar.slider(f"{mother_haemoglobin_col} Range", min_value=float(df[mother_haemoglobin_col].min()), max_value=float(df[mother_haemoglobin_col].max()), value=(float(df[mother_haemoglobin_col].min()), float(df[mother_haemoglobin_col].max())))
    cord_haemoglobin_range = st.sidebar.slider(f"{cord_haemoglobin_col} Range", min_value=float(df[cord_haemoglobin_col].min()), max_value=float(df[cord_haemoglobin_col].max()), value=(float(df[cord_haemoglobin_col].min()), float(df[cord_haemoglobin_col].max())))
    head_circum_day1_range = st.sidebar.slider(f"{head_circum_day1_col} Range", min_value=float(df[head_circum_day1_col].min()), max_value=float(df[head_circum_day1_col].max()), value=(float(df[head_circum_day1_col].min()), float(df[head_circum_day1_col].max())))
    weight_range = st.sidebar.slider(f"{weight_col} Range (kg)", min_value=float(df[weight_col].min()), max_value=float(df[weight_col].max()), value=(float(df[weight_col].min()), float(df[weight_col].max())))
    selected_blood_groups = st.sidebar.multiselect(f"Select {blood_group_col}", options=df[blood_group_col].unique(), default=df[blood_group_col].unique())
    selected_sex = st.sidebar.multiselect(f"Select {sex_col}", options=df[sex_col].unique(), default=df[sex_col].unique())

    # Apply filters
    filtered_df = df[
        (df[mother_haemoglobin_col].between(mother_haemoglobin_range[0], mother_haemoglobin_range[1])) &
        (df[cord_haemoglobin_col].between(cord_haemoglobin_range[0], cord_haemoglobin_range[1])) &
        (df[head_circum_day1_col].between(head_circum_day1_range[0], head_circum_day1_range[1])) &
        (df[weight_col].between(weight_range[0], weight_range[1])) &
        (df[blood_group_col].isin(selected_blood_groups)) &
        (df[sex_col].isin(selected_sex))
    ]

    # Define 'Group By' options
    possible_group_by_cols = [
        mother_haemoglobin_col, blood_group_col, cord_haemoglobin_col,
        head_circum_day1_col, head_circum_day3_col, length_col, weight_col,
        sex_col, address_col, parity_col
    ]
    available_group_by_cols = [col for col in possible_group_by_cols if col in df.columns]

    # Chart customization
    st.sidebar.header("Chart Customization")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Pie Chart", "Donut Chart", "Bar Chart", "Scatter Plot"], index=0)  # Pie Chart default
    group_by = st.sidebar.selectbox("Group By", available_group_by_cols)

    # Check if 'Group By' is numerical
    is_numerical_group_by = pd.api.types.is_numeric_dtype(df[group_by])

    # Binning for numerical 'Group By'
    if is_numerical_group_by:
        if chart_type in ["Pie Chart", "Donut Chart"]:
            num_bins = st.sidebar.slider(f"Number of bins for {group_by}", min_value=2, max_value=20, value=5)
            bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
            filtered_df['binned_group_by'] = pd.cut(filtered_df[group_by], bins=num_bins, labels=bin_labels)
            group_by_col = 'binned_group_by'
        elif chart_type == "Bar Chart":
            bin_option = st.sidebar.checkbox(f"Bin {group_by}", value=True)
            if bin_option:
                num_bins = st.sidebar.slider(f"Number of bins for {group_by}", min_value=2, max_value=20, value=5)
                bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                filtered_df['binned_group_by'] = pd.cut(filtered_df[group_by], bins=num_bins, labels=bin_labels)
                group_by_col = 'binned_group_by'
            else:
                group_by_col = group_by
        elif chart_type == "Scatter Plot":
            bin_option = st.sidebar.checkbox(f"Bin {group_by} for discrete colors", value=False)
            if bin_option:
                num_bins = st.sidebar.slider(f"Number of bins for {group_by}", min_value=2, max_value=20, value=5)
                bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                filtered_df['binned_group_by'] = pd.cut(filtered_df[group_by], bins=num_bins, labels=bin_labels)
                group_by_col = 'binned_group_by'
            else:
                group_by_col = group_by
    else:
        group_by_col = group_by

    color_scheme = st.sidebar.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma"])

    # Additional chart options
    if chart_type in ["Pie Chart", "Donut Chart"]:
        show_labels = st.sidebar.checkbox("Show Labels", value=True)
        show_percent = st.sidebar.checkbox("Show Percentages", value=True)
    elif chart_type == "Bar Chart":
        orientation = st.sidebar.selectbox("Orientation", ["vertical", "horizontal"])
    elif chart_type == "Scatter Plot":
        x_axis = st.sidebar.selectbox("X-Axis", available_group_by_cols)
        y_axis = st.sidebar.selectbox("Y-Axis", available_group_by_cols, index=1)

    # Color mapping
    color_map = {
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma,
        "Inferno": px.colors.sequential.Inferno,
        "Magma": px.colors.sequential.Magma
    }
    colors = color_map.get(color_scheme, px.colors.sequential.Viridis)

    # Generate chart
    if filtered_df.empty:
        st.warning("No data available with the selected filters.")
    else:
        if chart_type in ["Pie Chart", "Donut Chart", "Bar Chart"]:
            agg_df = filtered_df.groupby(group_by_col).size().reset_index(name='Count')
            if chart_type == "Pie Chart":
                fig = go.Figure(data=[go.Pie(
                    labels=agg_df[group_by_col],
                    values=agg_df['Count'],
                    textinfo='label+percent' if show_labels and show_percent else 'label' if show_labels else 'percent' if show_percent else None,
                    marker=dict(colors=colors)
                )])
            elif chart_type == "Donut Chart":
                fig = go.Figure(data=[go.Pie(
                    labels=agg_df[group_by_col],
                    values=agg_df['Count'],
                    textinfo='label+percent' if show_labels and show_percent else 'label' if show_labels else 'percent' if show_percent else None,
                    marker=dict(colors=colors),
                    hole=0.4
                )])
            elif chart_type == "Bar Chart":
                if orientation == "horizontal":
                    fig = px.bar(agg_df, y=group_by_col, x='Count', color=group_by_col, color_discrete_sequence=colors)
                else:
                    fig = px.bar(agg_df, x=group_by_col, y='Count', color=group_by_col, color_discrete_sequence=colors)
        else:  # Scatter Plot
            if is_numerical_group_by and not bin_option:
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=group_by_col, color_continuous_scale=colors)
            else:
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=group_by_col, color_discrete_sequence=colors)

        # Update layout
        fig.update_layout(title=f"{chart_type} by {group_by}", showlegend=True, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Display data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)
else:
    st.info("Please upload a CSV file to begin.")

st.markdown("**Note**: Run this app using `streamlit run app.py`.")

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

    # Check if required columns exist
    required_cols = [mother_haemoglobin_col, blood_group_col, cord_haemoglobin_col, head_circum_day1_col, head_circum_day3_col, length_col, weight_col, sex_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # Clean and preprocess data
    df[mother_haemoglobin_col] = pd.to_numeric(df[mother_haemoglobin_col], errors='coerce')
    df[cord_haemoglobin_col] = pd.to_numeric(df[cord_haemoglobin_col], errors='coerce')
    df[head_circum_day1_col] = pd.to_numeric(df[head_circum_day1_col], errors='coerce')
    df[head_circum_day3_col] = pd.to_numeric(df[head_circum_day3_col], errors='coerce')
    df[length_col] = pd.to_numeric(df[length_col], errors='coerce')
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')

    # Sidebar for filters
    st.sidebar.header("Filter Options")
    mother_haemoglobin_range = st.sidebar.slider(f"{mother_haemoglobin_col} Range", min_value=float(df[mother_haemoglobin_col].min()), max_value=float(df[mother_haemoglobin_col].max()), value=(float(df[mother_haemoglobin_col].min()), float(df[mother_haemoglobin_col].max())))
    cord_haemoglobin_range = st.sidebar.slider(f"{cord_haemoglobin_col} Range", min_value=float(df[cord_haemoglobin_col].min()), max_value=float(df[cord_haemoglobin_col].max()), value=(float(df[cord_haemoglobin_col].min()), float(df[cord_haemoglobin_col].max())))
    head_circum_day1_range = st.sidebar.slider(f"{head_circum_day1_col} Range", min_value=float(df[head_circum_day1_col].min()), max_value=float(df[head_circum_day1_col].max()), value=(float(df[head_circum_day1_col].min()), float(df[head_circum_day1_col].max())))
    weight_range = st.sidebar.slider(f"{weight_col} Range (kg)", min_value=float(df[weight_col].min()), max_value=float(df[weight_col].max()), value=(float(df[weight_col].min()), float(df[weight_col].max())))
    selected_blood_groups = st.sidebar.multiselect(f"Select {blood_group_col}", options=df[blood_group_col].unique(), default=df[blood_group_col].unique())
    selected_sex = st.sidebar.multiselect(f"Select {sex_col}", options=df[sex_col].unique(), default=df[sex_col].unique())

    # Filter data
    filtered_df = df[
        (df[mother_haemoglobin_col].between(mother_haemoglobin_range[0], mother_haemoglobin_range[1])) &
        (df[cord_haemoglobin_col].between(cord_haemoglobin_range[0], cord_haemoglobin_range[1])) &
        (df[head_circum_day1_col].between(head_circum_day1_range[0], head_circum_day1_range[1])) &
        (df[weight_col].between(weight_range[0], weight_range[1])) &
        (df[blood_group_col].isin(selected_blood_groups)) &
        (df[sex_col].isin(selected_sex))
    ]

    # Customization options
    st.sidebar.header("Chart Customization")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Pie Chart", "Donut Chart", "Bar Chart", "Scatter Plot"])
    group_by = st.sidebar.selectbox("Group By", [blood_group_col, sex_col])
    x_axis = st.sidebar.selectbox("X-Axis (Scatter Plot)", [mother_haemoglobin_col, cord_haemoglobin_col, head_circum_day1_col, head_circum_day3_col, length_col, weight_col])
    y_axis = st.sidebar.selectbox("Y-Axis (Scatter Plot)", [mother_haemoglobin_col, cord_haemoglobin_col, head_circum_day1_col, head_circum_day3_col, length_col, weight_col], index=1)
    color_scheme = st.sidebar.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma"])
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_percent = st.sidebar.checkbox("Show Percentages", value=True)
    explode_slice = st.sidebar.slider("Explode Slice (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    # Define color scale mapping
    color_map = {
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma,
        "Inferno": px.colors.sequential.Inferno,
        "Magma": px.colors.sequential.Magma
    }
    colors = color_map.get(color_scheme, px.colors.sequential.Viridis)

    # Main content
    if filtered_df.empty:
        st.warning("No data available with the selected filters.")
    else:
        # Aggregate data for Pie/Bar/Donut
        agg_df = filtered_df.groupby(group_by).size().reset_index(name='Count')

        # Create chart
        if chart_type in ["Pie Chart", "Donut Chart"]:
            fig = go.Figure(data=[
                go.Pie(
                    labels=agg_df[group_by],
                    values=agg_df['Count'],
                    textinfo='label+percent' if show_labels and show_percent else 'label' if show_labels else 'percent' if show_percent else None,
                    marker=dict(colors=colors),
                    pull=[explode_slice if i == 0 else 0 for i in range(len(agg_df))],
                    hole=0.4 if chart_type == "Donut Chart" else 0
                )
            ])
        elif chart_type == "Bar Chart":
            fig = px.bar(
                agg_df,
                x=group_by,
                y='Count',
                color=group_by,
                color_discrete_sequence=colors,
                labels={'Count': 'Number of Records'}
            )
        else:  # Scatter Plot
            fig = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color=group_by,
                size=weight_col,
                color_discrete_sequence=colors,
                labels={x_axis: x_axis, y_axis: y_axis}
            )

        # Update layout
        fig.update_layout(
            title=f"{chart_type} by {group_by}",
            showlegend=True,
            template="plotly_dark" if st.sidebar.checkbox("Dark Theme", value=False) else "plotly",
            height=600
        )

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

        # Display filtered data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)
else:
    st.info("Please upload a CSV file to begin.")

# Run instructions
st.markdown("**Note**: Run this app using `streamlit run app.py` in your terminal.")
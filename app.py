import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(page_title="ICMR Data Visualization", layout="wide")

# Title and description
st.title("ICMR Maternal and Newborn Data Dashboard")
st.markdown("Upload your CSV to explore Haemoglobin, Blood Group, Cord Blood, Head Circumference, Length, and Weight with enhanced pie charts.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

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
    required_cols = [mother_haemoglobin_col, blood_group_col, cord_haemoglobin_col, head_circum_day1_col, head_circum_day3_col, length_col, weight_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # Standardize categorical columns
    df[sex_col] = df[sex_col].str.lower().str.strip().replace({
        'm': 'male', 'f': 'female', 'o': 'other', 'u': 'unknown',
        'boy': 'male', 'girl': 'female', 'not specified': 'unknown'
    }).fillna('unknown') if sex_col in df.columns else pd.Series('unknown', index=df.index)
    df[blood_group_col] = df[blood_group_col].str.strip().str.upper()

    # Clean numerical data
    numerical_cols = [mother_haemoglobin_col, cord_haemoglobin_col, head_circum_day1_col, head_circum_day3_col, length_col, weight_col]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filter Options")
    mother_haemoglobin_range = st.sidebar.slider(
        f"{mother_haemoglobin_col} Range",
        min_value=float(df[mother_haemoglobin_col].min()),
        max_value=float(df[mother_haemoglobin_col].max()),
        value=(float(df[mother_haemoglobin_col].min()), float(df[mother_haemoglobin_col].max())),
        step=0.1
    )
    cord_haemoglobin_range = st.sidebar.slider(
        f"{cord_haemoglobin_col} Range",
        min_value=float(df[cord_haemoglobin_col].min()),
        max_value=float(df[cord_haemoglobin_col].max()),
        value=(float(df[cord_haemoglobin_col].min()), float(df[cord_haemoglobin_col].max())),
        step=0.1
    )
    head_circum_day1_range = st.sidebar.slider(
        f"{head_circum_day1_col} Range",
        min_value=float(df[head_circum_day1_col].min()),
        max_value=float(df[head_circum_day1_col].max()),
        value=(float(df[head_circum_day1_col].min()), float(df[head_circum_day1_col].max())),
        step=0.1
    )
    weight_range = st.sidebar.slider(
        f"{weight_col} Range (kg)",
        min_value=float(df[weight_col].min()),
        max_value=float(df[weight_col].max()),
        value=(float(df[weight_col].min()), float(df[weight_col].max())),
        step=0.01
    )
    selected_blood_groups = st.sidebar.multiselect(
        f"Select {blood_group_col}",
        options=sorted(df[blood_group_col].dropna().unique()),
        default=sorted(df[blood_group_col].dropna().unique())
    )
    gender_options = ['male', 'female', 'other', 'unknown']
    selected_sex = st.sidebar.multiselect(
        f"Select {sex_col}",
        options=gender_options,
        default=gender_options
    ) if sex_col in df.columns else gender_options

    # Apply filters
    filtered_df = df[
        (df[mother_haemoglobin_col].between(mother_haemoglobin_range[0], mother_haemoglobin_range[1], inclusive='both')) &
        (df[cord_haemoglobin_col].between(cord_haemoglobin_range[0], cord_haemoglobin_range[1], inclusive='both')) &
        (df[head_circum_day1_col].between(head_circum_day1_range[0], head_circum_day1_range[1], inclusive='both')) &
        (df[weight_col].between(weight_range[0], weight_range[1], inclusive='both')) &
        (df[blood_group_col].isin(selected_blood_groups)) &
        (df[sex_col].isin(selected_sex) if sex_col in df.columns else True)
    ]

    # Define compulsory 'Group By' columns
    compulsory_group_by_cols = [
        mother_haemoglobin_col, blood_group_col, cord_haemoglobin_col,
        head_circum_day1_col, head_circum_day3_col, length_col, weight_col
    ]
    optional_group_by_cols = [col for col in [sex_col, address_col, parity_col] if col in df.columns]
    available_group_by_cols = compulsory_group_by_cols + optional_group_by_cols

    # Chart customization
    st.sidebar.header("Chart Customization")
    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        ["Pie Chart", "Donut Chart", "Bar Chart", "Scatter Plot"],
        index=0  # Default to Pie Chart
    )
    group_by_cols = st.sidebar.multiselect(
        "Group By (select multiple for combined grouping)",
        options=available_group_by_cols,
        default=[weight_col, sex_col] if sex_col in df.columns else [weight_col]
    )

    if not group_by_cols:
        st.error("Please select at least one 'Group By' column.")
        st.stop()

    # Generate range options for numerical columns
    range_options = {}
    for col in numerical_cols:
        min_val = filtered_df[col].min()
        max_val = filtered_df[col].max()
        if pd.isna(min_val) or pd.isna(max_val):
            range_options[col] = []
            continue
        if col == mother_haemoglobin_col:
            edges = [6.3, 8.5, 10.7, 13.7, 15.4]
        elif col == cord_haemoglobin_col:
            edges = [7.1, 11.3, 14.6, 21.2]
        elif col in [head_circum_day1_col, head_circum_day3_col]:
            edges = [32.8, 34.5, 35.5, 36.5]
        elif col == length_col:
            edges = [41.0, 45.0, 48.0, 51.0]
        elif col == weight_col:
            edges = [2.0, 2.5, 3.0, 3.7]
        else:
            step = (max_val - min_val) / 5
            edges = np.arange(min_val, max_val + step, step)
            edges = [round(e, 2) for e in edges]
        ranges = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)]
        range_options[col] = ranges

    # Binning for numerical 'Group By' columns
    group_by_col_mappings = {}
    for col in group_by_cols:
        if col in numerical_cols:
            st.sidebar.write(f"{col} is numerical. Select ranges to include.")
            selected_ranges = st.sidebar.multiselect(
                f"Select ranges for {col}",
                options=range_options.get(col, []),
                default=range_options.get(col, [])[:3]
            )
            if not selected_ranges:
                st.error(f"Please select at least one range for {col}.")
                st.stop()
            try:
                bin_edges = []
                bin_labels = []
                sorted_ranges = sorted(selected_ranges, key=lambda x: float(x.split('-')[0]))
                for r in sorted_ranges:
                    start, end = map(float, r.split('-'))
                    bin_edges.append(start)
                    bin_labels.append(r)
                bin_edges.append(end)
                min_val = filtered_df[col].min()
                max_val = filtered_df[col].max()
                if pd.notna(min_val) and min_val < bin_edges[0]:
                    bin_edges.insert(0, min_val - 0.01)
                    bin_labels.insert(0, f"{min_val:.2f}-{bin_edges[1]:.2f}")
                if pd.notna(max_val) and max_val > bin_edges[-1]:
                    bin_edges.append(max_val + 0.01)
                    bin_labels.append(f"{bin_edges[-2]:.2f}-{max_val:.2f}")
                filtered_df[f'binned_{col}'] = pd.cut(
                    filtered_df[col],
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True,
                    right=True
                )
                group_by_col_mappings[col] = f'binned_{col}'
            except Exception as e:
                st.error(f"Error binning {col}: {e}")
                st.stop()
        else:
            group_by_col_mappings[col] = col

    # Create combined group column
    group_by_cols_mapped = [group_by_col_mappings[col] for col in group_by_cols]
    valid_group_by_cols_mapped = [col for col in group_by_cols_mapped if col in filtered_df.columns]
    if not valid_group_by_cols_mapped:
        st.error("No valid 'Group By' columns after binning. Please check selected ranges.")
        st.stop()
    filtered_df['combined_group'] = filtered_df[valid_group_by_cols_mapped].astype(str).agg('_'.join, axis=1)
    group_by_col = 'combined_group'

    # Additional customization options
    color_scheme = st.sidebar.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma"])
    theme = st.sidebar.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    font_size = st.sidebar.slider("Font Size", min_value=8, max_value=24, value=12)
    legend_position = st.sidebar.selectbox("Legend Position", ["top", "bottom", "left", "right"])
    chart_title = st.sidebar.text_input("Chart Title", value=f"{chart_type} by {', '.join(group_by_cols)}")

    # Chart-specific options
    sort_by_value = st.sidebar.checkbox("Sort by Value", value=False)
    if chart_type in ["Pie Chart", "Donut Chart"]:
        text_position = st.sidebar.selectbox("Text Position", ["inside", "outside", "auto"], index=2)
        label_font_size = st.sidebar.slider("Label Font Size", min_value=8, max_value=20, value=12)
        label_font_color = st.sidebar.color_picker("Label Font Color", value="#FFFFFF")
        opacity = st.sidebar.slider("Slice Opacity (0-1)", min_value=0.5, max_value=1.0, value=0.9, step=0.1)
        use_3d = st.sidebar.checkbox("3D Effect", value=False)
        border_width = st.sidebar.slider("Slice Border Width", min_value=0, max_value=5, value=1)
        border_color = st.sidebar.color_picker("Slice Border Color", value="#000000")
        show_tooltips = st.sidebar.checkbox("Show Tooltips", value=True)
        explode_slice = st.sidebar.slider("Explode Slice (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        rotation_angle = st.sidebar.slider("Rotation Angle (degrees)", min_value=0, max_value=360, value=0)
    elif chart_type == "Bar Chart":
        orientation = st.sidebar.selectbox("Orientation", ["vertical", "horizontal"])
        bar_width = st.sidebar.slider("Bar Width", min_value=0.1, max_value=1.0, value=0.8)
        show_grid = st.sidebar.checkbox("Show Grid", value=False)
        show_tooltips = st.sidebar.checkbox("Show Tooltips", value=True)
    elif chart_type == "Scatter Plot":
        x_axis = st.sidebar.selectbox("X-Axis", available_group_by_cols)
        y_axis = st.sidebar.selectbox("Y-Axis", available_group_by_cols, index=1)
        marker_size = st.sidebar.slider("Marker Size", min_value=5, max_value=20, value=10)
        marker_symbol = st.sidebar.selectbox("Marker Symbol", ["circle", "square", "diamond", "cross", "x"], index=0)
        add_trendline = st.sidebar.checkbox("Add Trendline", value=False)
        show_grid = st.sidebar.checkbox("Show Grid", value=False)
        show_tooltips = st.sidebar.checkbox("Show Tooltips", value=True)

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
            agg_df = filtered_df.groupby(group_by_col, observed=True).size().reset_index(name='Count')
            if sort_by_value:
                agg_df = agg_df.sort_values('Count', ascending=False)
            if chart_type in ["Pie Chart", "Donut Chart"]:
                num_slices = len(agg_df)
                pie_size = min(800, 400 + num_slices * 20)
                fig = go.Figure(data=[
                    go.Pie(
                        labels=agg_df[group_by_col],
                        values=agg_df['Count'],
                        textinfo='label+percent',
                        textposition=text_position,
                        textfont=dict(size=label_font_size, color=label_font_color),
                        marker=dict(
                            colors=colors,
                            line=dict(color=border_color, width=border_width)
                        ),
                        pull=[explode_slice if i == 0 else 0 for i in range(num_slices)],
                        rotation=rotation_angle,
                        hoverinfo='label+percent+value' if show_tooltips else 'none',
                        hole=0.4 if chart_type == "Donut Chart" else 0,
                        opacity=opacity,
                        showlegend=True
                    )
                ])
                if use_3d:
                    fig.update_layout(
                        scene=dict(
                            aspectmode="cube",
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(visible=False)
                        ),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                fig.update_traces(opacity=opacity)
            elif chart_type == "Bar Chart":
                if orientation == "horizontal":
                    fig = px.bar(
                        agg_df,
                        y=group_by_col,
                        x='Count',
                        color=group_by_col,
                        color_discrete_sequence=colors,
                        labels={'Count': 'Number of Records'},
                        orientation='h'
                    )
                    fig.update_traces(width=bar_width)
                else:
                    fig = px.bar(
                        agg_df,
                        x=group_by_col,
                        y='Count',
                        color=group_by_col,
                        color_discrete_sequence=colors,
                        labels={'Count': 'Number of Records'}
                    )
                    fig.update_traces(width=bar_width)
                fig.update_layout(
                    showlegend=True,
                    xaxis=dict(showgrid=show_grid),
                    yaxis=dict(showgrid=show_grid),
                    hovermode='closest' if show_tooltips else False
                )
        else:  # Scatter Plot
            if len(group_by_cols) > 1:
                color_col = group_by_cols[0]
                size_col = group_by_cols[1]
            else:
                color_col = group_by_cols[0]
                size_col = weight_col
            hover_data_cols = [col for col in group_by_cols_mapped if col in filtered_df.columns]
            try:
                fig = px.scatter(
                    filtered_df,
                    x=x_axis,
                    y=y_axis,
                    color=group_by_col_mappings.get(color_col, color_col),
                    size=filtered_df[group_by_col_mappings.get(size_col, size_col)].fillna(
                        filtered_df[size_col].mean() if size_col in filtered_df.columns else 1
                    ),
                    size_max=marker_size,
                    symbol=marker_symbol,
                    color_discrete_sequence=colors if not pd.api.types.is_numeric_dtype(filtered_df[group_by_col_mappings.get(color_col, color_col)]) else None,
                    color_continuous_scale=colors if pd.api.types.is_numeric_dtype(filtered_df[group_by_col_mappings.get(color_col, color_col)]) else None,
                    trendline="ols" if add_trendline else None,
                    labels={x_axis: x_axis, y_axis: y_axis},
                    hover_data=hover_data_cols if show_tooltips and hover_data_cols else None
                )
            except ValueError as e:
                st.error(f"Error generating scatter plot: {e}. Ensure selected columns contain valid numerical data.")
                st.stop()
            fig.update_layout(
                showlegend=True,
                xaxis=dict(showgrid=show_grid),
                yaxis=dict(showgrid=show_grid),
                hovermode='closest' if show_tooltips else False
            )

        # Update layout
        fig.update_layout(
            title=chart_title,
            showlegend=True,
            height=pie_size if chart_type in ["Pie Chart", "Donut Chart"] else 600,
            template=theme,
            font=dict(size=font_size),
            legend=dict(
                orientation="h" if legend_position in ["top", "bottom"] else "v",
                yanchor="top" if legend_position == "bottom" else "bottom" if legend_position == "top" else "middle",
                xanchor="center" if legend_position in ["top", "bottom"] else "left" if legend_position == "right" else "right",
                y=1.1 if legend_position == "top" else -0.1 if legend_position == "bottom" else 0.5,
                x=0.5 if legend_position in ["top", "bottom"] else 1.05 if legend_position == "right" else -0.05
            )
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

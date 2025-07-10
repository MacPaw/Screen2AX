"""
This is a Streamlit app for visualizing accessibility metadata.

To run the app, use the following command:
streamlit run visualiser_app.py

Also, please select dark theme in the settings.
"""


import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import plotly.express as px

class StreamlitAccessibilityVisualizer:
    def __init__(self, image_path, custom_json_path, system_json_path, system_scale_factor=2.0):
        """
        Initialize the accessibility visualizer with file paths
        
        Parameters:
        - image_path: Path to the app screenshot image
        - custom_json_path: Path to the custom accessibility JSON file
        - system_json_path: Path to the system accessibility JSON file
        - system_scale_factor: Scale factor to apply to system data coordinates (default: 2.0)
        """
        self.image_path = image_path
        self.custom_json_path = custom_json_path
        self.system_json_path = system_json_path
        self.custom_data = None
        self.system_data = None
        self.elements = {'custom': [], 'system': []}
        self.fig = None
        self.system_scale_factor = system_scale_factor
        
        # Load data
        self.load_data()
        self.extract_elements()
    
    def load_data(self):
        """Load the JSON data and image"""
        # Load custom JSON
        with open(self.custom_json_path, 'r') as f:
            self.custom_data = json.load(f)
        
        # Load system JSON
        with open(self.system_json_path, 'r') as f:
            self.system_data = json.load(f)
        
        # Load image
        self.image = Image.open(self.image_path)
        self.img_width, self.img_height = self.image.size
    
    def extract_elements_from_custom(self, node, elements, parent=None, depth=0):
        """Recursively extract elements with bounding boxes from custom data"""
        # Extract information for this node if it has a bounding box
        if 'box' in node and len(node['box']) == 4:
            x1, y1, x2, y2 = node['box']
            width = x2 - x1
            height = y2 - y1
            
            # Only include elements with non-zero width and height
            if width > 0 and height > 0:
                element = {
                    'x0': x1,
                    'y0': y1,
                    'x1': x2,
                    'y1': y2,
                    'width': width,
                    'height': height,
                    'value': node.get('value', None),
                    'cls': node.get('cls', None),
                    'depth': depth
                }
                elements.append(element)
        
        # Process children recursively
        if 'children' in node and node['children']:
            for child in node['children']:
                self.extract_elements_from_custom(child, elements, node, depth + 1)
    
    def extract_elements_from_system(self, node, elements, parent=None, depth=0):
        """Recursively extract elements with position and size from system data"""
        # Extract information for this node if it has position and size
        if 'position' in node and 'size' in node:
            try:
                pos = node['position'].split(';')
                size = node['size'].split(';')
                
                if len(pos) == 2 and len(size) == 2:
                    # Apply scaling factor to adjust system coordinates to match the screenshot scale
                    x = float(pos[0]) * self.system_scale_factor
                    y = float(pos[1]) * self.system_scale_factor
                    width = float(size[0]) * self.system_scale_factor
                    height = float(size[1]) * self.system_scale_factor
                    
                    # Only include elements with non-zero width and height
                    if width > 0 and height > 0:
                        element = {
                            'x0': x,
                            'y0': y,
                            'x1': x + width,
                            'y1': y + height,
                            'width': width,
                            'height': height,
                            'value': node.get('value', None),
                            'name': node.get('name', None),
                            'role': node.get('role', None),
                            'description': node.get('description', None),
                            'depth': depth
                        }
                        elements.append(element)
            except (ValueError, IndexError):
                pass  # Skip if conversion fails
        
        # Process children recursively
        if 'children' in node and node['children']:
            for child in node['children']:
                self.extract_elements_from_system(child, elements, node, depth + 1)
    
    def extract_elements(self):
        """Extract elements from both data sources"""
        # Extract from custom data
        self.extract_elements_from_custom(self.custom_data, self.elements['custom'])
        
        # Extract from system data
        self.extract_elements_from_system(self.system_data, self.elements['system'])
        
        print(f"Extracted {len(self.elements['custom'])} elements from custom data")
        print(f"Extracted {len(self.elements['system'])} elements from system data")
    
    def create_figure_config(self):
        """Create a configuration object for the Plotly figure"""
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['toggleHover'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'accessibility_visualization',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    
    def create_plotly_figure(self, data_source, element_types=None, min_size=0, max_depth=None):
        """
        Create a plotly figure with the app screenshot as background
        
        Parameters:
        - data_source: 'custom' or 'system'
        - element_types: List of element types to include (cls for custom, role for system)
        - min_size: Minimum element size (width*height) to include
        - max_depth: Maximum depth of elements to include
        
        Returns:
        - fig: Plotly figure
        """
        # Create figure
        self.fig = go.Figure()
        
        # Calculate aspect ratio and set figure size
        aspect_ratio = self.img_height / self.img_width
        display_width = min(1200, self.img_width)  # Limit max width for large images
        display_height = int(display_width * aspect_ratio)
        
        # Add the screenshot as a background image
        self.fig.add_layout_image(
            dict(
                source=self.image,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=self.img_width,
                sizey=self.img_height,
                sizing="contain",  # "contain" to preserve aspect ratio
                opacity=1,
                layer="below"
            )
        )
        
        # Set axes properties
        self.fig.update_xaxes(
            range=[0, self.img_width],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="y",  # Lock the aspect ratio
            scaleratio=1      # 1:1 aspect ratio
        )
        
        self.fig.update_yaxes(
            range=[self.img_height, 0],  # Inverted y-axis to match image coordinates
            showticklabels=False,
            showgrid=False,
            zeroline=False
        )
        
        # Update layout with a dark theme
        self.fig.update_layout(
            title=f"Accessibility Visualization ({data_source.capitalize()} Data)",
            title_font=dict(size=20, color="white"),
            autosize=False,  # Use fixed size instead of autosize
            width=display_width,
            height=display_height,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode="closest",
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color="white"),
            legend=dict(
                title_font=dict(size=14),
                font=dict(size=12),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                itemsizing='constant',  # Make legend items all the same size
                itemwidth=30,
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='right',
                x=1.1,
                tracegroupgap=5
            ),
            modebar=dict(
                bgcolor="rgba(0,0,0,0)",
                color="white",
                activecolor="#636EFA"
            ),
            dragmode="pan",  # Set default interaction mode to pan
        )
        
        return self.fig
    
    def add_bounding_boxes(self, data_source, element_types=None, min_size=0, max_depth=None):
        """
        Add bounding boxes to the plotly figure
        
        Parameters:
        - data_source: 'custom' or 'system'
        - element_types: List of element types to include (cls for custom, role for system)
        - min_size: Minimum element size (width*height) to include
        - max_depth: Maximum depth of elements to include
        """
        # Get elements for the current data source
        current_elements = self.elements[data_source]
        
        # Apply filters
        filtered_elements = current_elements
        
        # Filter by element type
        if element_types and len(element_types) > 0:
            if data_source == 'custom':
                filtered_elements = [e for e in filtered_elements if e.get('cls') in element_types]
            else:  # system
                filtered_elements = [e for e in filtered_elements if e.get('role') in element_types]
        
        # Filter by minimum size
        if min_size > 0:
            filtered_elements = [e for e in filtered_elements if e.get('width', 0) * e.get('height', 0) >= min_size]
        
        # Filter by maximum depth
        if max_depth is not None:
            filtered_elements = [e for e in filtered_elements if e.get('depth', 0) <= max_depth]
        
        # Sort elements by depth (deepest first, to have shallower elements on top)
        sorted_elements = sorted(filtered_elements, key=lambda e: e.get('depth', 0), reverse=True)
        
        # Get element types for the current data source
        if data_source == 'custom':
            all_element_types = sorted(set(e.get('cls', 'Unknown') for e in current_elements))
        else:  # system
            all_element_types = sorted(set(e.get('role', 'Unknown') for e in current_elements))
        
        # Create a colormap
        colors = px.colors.qualitative.Plotly  # Use Plotly's built-in color scale
        color_map = {elem_type: colors[i % len(colors)] for i, elem_type in enumerate(all_element_types)}
        
        # Group elements by type and add them to the figure
        for elem_type in all_element_types:
            if element_types and elem_type not in element_types:
                continue
                
            color = color_map[elem_type]
            
            # Create lists for shape data
            x0_list, y0_list, x1_list, y1_list = [], [], [], []
            hover_texts = []
            
            # Create separate lists for Group elements (which will not have hover)
            group_x0_list, group_y0_list, group_x1_list, group_y1_list = [], [], [], []
            
            # Filter elements by type and collect data
            for elem in sorted_elements:
                if ((data_source == 'custom' and elem.get('cls') == elem_type) or
                    (data_source == 'system' and elem.get('role') == elem_type)):
                    
                    # Skip very large elements that might be the background
                    width = elem['width'] if 'width' in elem else elem['x1'] - elem['x0']
                    height = elem['height'] if 'height' in elem else elem['y1'] - elem['y0']
                    if width > self.img_width * 0.95 and height > self.img_height * 0.95:
                        continue
                    
                    # Check if this is a Group element
                    is_group = False
                    if (data_source == 'custom' and elem.get('cls') == 'Group') or \
                       (data_source == 'system' and elem.get('role') == 'AXGroup'):
                        is_group = True
                        group_x0_list.append(elem['x0'])
                        group_y0_list.append(elem['y0'])
                        group_x1_list.append(elem['x1'])
                        group_y1_list.append(elem['y1'])
                    else:
                        # Only add non-Group elements to the hoverable list
                        x0_list.append(elem['x0'])
                        y0_list.append(elem['y0'])
                        x1_list.append(elem['x1'])
                        y1_list.append(elem['y1'])
                        
                        # Create simplified hover text with just the value (if available)
                        # Otherwise use type/role
                        if data_source == 'custom':
                            hover_text = elem.get('value', elem.get('cls', ''))
                        else:  # system
                            hover_text = elem.get('value', elem.get('name', elem.get('role', '')))
                        
                        # Make sure we have a value to show
                        if not hover_text:
                            if data_source == 'custom':
                                hover_text = elem.get('cls', 'Unknown')
                            else:
                                hover_text = elem.get('role', 'Unknown')
                        
                        hover_texts.append(hover_text)
            
            # Add shapes if we have any for this type
            if x0_list:
                # Add hoverable elements (non-Group elements)
                self.fig.add_trace(
                    go.Scatter(
                        x=[(x0 + x1) / 2 for x0, x1 in zip(x0_list, x1_list)],  # Center point for hover
                        y=[(y0 + y1) / 2 for y0, y1 in zip(y0_list, y1_list)],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color='rgba(0,0,0,0)'  # Invisible markers, just for hover
                        ),
                        hoverinfo='text',
                        hovertemplate="<span style='font-size: 16px;'>%{hovertext}</span><extra></extra>",
                        hovertext=hover_texts,
                        name=elem_type,
                        showlegend=True,
                        legendgroup=elem_type,
                        marker_color=color  # Set marker color to match the boxes (for legend)
                    )
                )
                
                # Add rectangle shapes for hoverable elements
                for i in range(len(x0_list)):
                    self.fig.add_shape(
                        type="rect",
                        x0=x0_list[i],
                        y0=y0_list[i],
                        x1=x1_list[i],
                        y1=y1_list[i],
                        line=dict(
                            color=color,
                            width=1,  # Thinner line
                        ),
                        fillcolor="rgba(0,0,0,0)",
                        opacity=0.9
                    )
            
            # Add Group elements (non-hoverable)
            if group_x0_list and elem_type in ['Group', 'AXGroup']:
                # We don't add a trace for these (so they won't be hoverable)
                # Just add the shapes
                for i in range(len(group_x0_list)):
                    self.fig.add_shape(
                        type="rect",
                        x0=group_x0_list[i],
                        y0=group_y0_list[i],
                        x1=group_x1_list[i],
                        y1=group_y1_list[i],
                        line=dict(
                            color=color,
                            width=1,  # Thinner line
                        ),
                        fillcolor="rgba(0,0,0,0)",
                        opacity=0.9
                    )
    
    def get_element_types(self, data_source):
        """Get all element types for a data source"""
        if data_source == 'custom':
            return sorted(set(e.get('cls', 'Unknown') for e in self.elements['custom']))
        else:  # system
            return sorted(set(e.get('role', 'Unknown') for e in self.elements['system']))
    
    def get_max_depth(self, data_source):
        """Get the maximum depth of elements for a data source"""
        if data_source == 'custom':
            return max((e.get('depth', 0) for e in self.elements['custom']), default=0)
        else:  # system
            return max((e.get('depth', 0) for e in self.elements['system']), default=0)


# Streamlit app
def main():
    # Configure the page with a dark theme and expanded layout
    st.set_page_config(
        layout="wide", 
        page_title="Accessibility Visualization",
        page_icon="🔍",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': 'https://github.com/yourusername/accessibility-visualizer',
            'Report a bug': 'https://github.com/yourusername/accessibility-visualizer/issues',
            'About': "Interactive visualization tool for accessibility metadata"
        }
    )
    
    # Custom CSS for dark theme and better spacing
    st.markdown("""
    <style>
    /* Dark theme adjustments */
    .stApp {
        background-color: #111111;
        color: #f0f0f0;
    }
    
    /* Control panel styling */
    .stButton button {
        background-color: #4e54c8;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton button:hover {
        background-color: #3a3fc0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: white;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    
    /* Divider styling */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #333333;
    }
    
    /* Frame the visualization */
    .element-container:has(iframe) {
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 5px;
        background-color: #1a1a1a;
    }
    
    /* Better dataframe styling */
    .dataframe-container {
        border-radius: 5px;
        background-color: #1a1a1a;
    }
    
    /* Improve radio button visibility */
    .stRadio label {
        color: #f0f0f0 !important;
    }
    
    /* Fix slider colors */
    div[data-testid="stThumbValue"] {
        color: white !important;
    }
    
    /* Remove blue line from sliders */
    .stSlider [data-baseweb="slider"] {
        border-top: none !important;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    # Main app title with icon
    st.markdown("""
    # 🔍 Interactive Accessibility Visualization
    Visualize and analyze accessibility metadata from custom and system sources.
    """)
    
    # File Selection (in a collapsible section)
    with st.expander("File Selection", expanded=False):
        # File upload controls in a horizontal layout
        upload_col1, upload_col2, upload_col3 = st.columns(3)
        
        with upload_col1:
            # Screenshot image upload
            uploaded_image = st.file_uploader("Upload Screenshot Image", type=["png", "jpg", "jpeg"])
        
        with upload_col2:
            # Custom JSON upload
            uploaded_custom_json = st.file_uploader("Upload Custom Accessibility JSON", type=["json"])
        
        with upload_col3:
            # System JSON upload
            uploaded_system_json = st.file_uploader("Upload System Accessibility JSON", type=["json"])
        
    
    # Always visible filters (not in collapsible section)
    # Create a 3-column layout for the always-visible filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Data source selection (custom vs system)
        data_source = st.radio("Data Source", ["custom", "system"], horizontal=True)
    
    # Set system_scale_factor to a default value (not displayed in UI)
    system_scale_factor = 2.0
    
    # Check if files are uploaded or use default files
    if uploaded_image is None or uploaded_custom_json is None or uploaded_system_json is None:
        # Display help information
        # Display instructions and help information
        st.markdown("### How to Use This Tool")
        st.markdown("""
        This tool visualizes accessibility metadata from both custom and system JSON files overlaid on an app screenshot.
        
        #### Getting Started:
        1. Click on "File Selection" to expand the upload section
        2. Upload your screenshot image (PNG, JPG)
        3. Upload your custom accessibility JSON file
        4. Upload your system accessibility JSON file
        
        #### Features:
        - **Switch Data Sources**: Toggle between custom and system accessibility data
        - **Filter Elements**: Show/hide specific element types
        - **Depth Control**: Filter elements by their depth in the hierarchy
        - **Interactive Hover**: Mouse over elements to see detailed information
        """)
        return
    else:
        # Save uploaded files temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        with open("temp_custom.json", "wb") as f:
            f.write(uploaded_custom_json.getbuffer())
        
        with open("temp_system.json", "wb") as f:
            f.write(uploaded_system_json.getbuffer())
        
        image_path = "temp_image.png"
        custom_json_path = "temp_custom.json"
        system_json_path = "temp_system.json"
    
    # Create visualizer instance with the fixed scale factor
    visualizer = StreamlitAccessibilityVisualizer(
        image_path, 
        custom_json_path, 
        system_json_path,
        system_scale_factor=system_scale_factor
    )
    
    # Get element types for the selected data source
    element_types = visualizer.get_element_types(data_source)
    
    # Add the remaining filters (element types and depth)
    with filter_col2:
        # Element type filter
        selected_element_types = st.multiselect(
            "Element types to display",
            options=element_types,
            default=element_types
        )
    
    # Set min_size to 0 (no minimum size filter)
    min_size = 0
    
    with filter_col3:
        # Depth filter
        max_depth = visualizer.get_max_depth(data_source)
        depth_filter = st.slider(
            "Maximum element depth",
            min_value=0,
            max_value=max_depth,
            value=max_depth,
            step=1
        )
    
    # Generate visualization
    fig = visualizer.create_plotly_figure(
        data_source=data_source,
        element_types=selected_element_types,
        min_size=min_size,
        max_depth=depth_filter
    )
    
    # Add bounding boxes and other elements
    visualizer.add_bounding_boxes(
        data_source=data_source,
        element_types=selected_element_types,
        min_size=min_size,
        max_depth=depth_filter
    )
    
    # Create config for the figure
    config = visualizer.create_figure_config()
    
    # Display the visualization in full width
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Add space between visualization and statistics
    st.markdown("---")
    
    # Create tabs for statistics and help information
    stats_tab, help_tab = st.tabs(["Element Statistics", "Help & Information"])
    
    with stats_tab:
        # Count elements by type
        if data_source == 'custom':
            df = pd.DataFrame([
                {"Type": e.get('cls', 'Unknown'), "Depth": e.get('depth', 0)}
                for e in visualizer.elements['custom']
                if e.get('width', 0) * e.get('height', 0) >= min_size and e.get('depth', 0) <= depth_filter
            ])
        else:  # system
            df = pd.DataFrame([
                {"Type": e.get('role', 'Unknown'), "Depth": e.get('depth', 0)}
                for e in visualizer.elements['system']
                if e.get('width', 0) * e.get('height', 0) >= min_size and e.get('depth', 0) <= depth_filter
            ])
        
        if not df.empty:
            type_counts = df['Type'].value_counts().reset_index()
            type_counts.columns = ['Element Type', 'Count']
            
            # Create three columns for the statistics
            chart_cols = st.columns(3)
            
            with chart_cols[0]:
                # Display counts table without index numbers
                st.subheader(f"Element Type Counts \n ({data_source.capitalize()})")
                st.dataframe(
                    type_counts[['Element Type', 'Count']].set_index('Element Type'), 
                    use_container_width=True
                )
            
            with chart_cols[1]:
                # Display pie chart of element types
                st.subheader(f"Distribution of Element Types \n ({data_source.capitalize()})")
                fig_pie = px.pie(
                    type_counts, 
                    values='Count', 
                    names='Element Type',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                # Improve pie chart appearance
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    margin=dict(t=30, b=0, l=0, r=0),
                    showlegend=False,
                    # legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor="#1a1a1a",
                    plot_bgcolor="#1a1a1a",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
            
            with chart_cols[2]:
                # Display depth histogram
                st.subheader(f"Element Depth Distribution \n ({data_source.capitalize()})")
                fig_hist = px.histogram(
                    df, 
                    x='Depth',
                    nbins=max_depth+1,
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist.update_layout(
                    xaxis_title="Depth",
                    yaxis_title="Count",
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="#1a1a1a",
                    plot_bgcolor="#1a1a1a",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No elements match the current filters.")
    
    with help_tab:
        st.markdown("""
        ## How to Use This Tool
        
        This interactive visualization tool helps you analyze accessibility metadata from both custom and system sources, overlaid on an application screenshot.
        
        ### Getting Started
        
        1. **Upload Files** (in the File Selection section):
           - Screenshot image (PNG, JPG)
           - Custom accessibility JSON file
           - System accessibility JSON file
        
        2. **Configure Visualization**:
           - Switch between custom and system data sources
           - Filter elements by type
           - Filter elements by hierarchy depth
        
        3. **Interact with Visualization**:
           - Hover over elements to see detailed information
           - Zoom in/out using the mousewheel
           - Pan by clicking and dragging
           - Use the toolbar to reset view, download as PNG, etc.
        
        ### Understanding the Data
        
        - **Custom Data**: Uses `box` attribute with [x1, y1, x2, y2] coordinates
        - **System Data**: Uses `position` (x;y) and `size` (width;height) attributes
        - **Element Types**: 
          - Custom: Group, Text, AXButton, AXImage, etc.
          - System: AXWindow, AXGroup, AXButton, etc.
        - **Depth**: Indicates the nesting level in the element hierarchy
        
        ### Element Information on Hover
        
        When hovering over elements, you'll see different information depending on the data source:
        
        - **Custom data**: Type, Value, Position, Size, Depth
        - **System data**: Role, Name, Value, Description, Position, Size, Depth
        """)
    
    # Clean up temporary files
    if uploaded_image is not None:
        try:
            os.remove("temp_image.png")
            os.remove("temp_custom.json")
            os.remove("temp_system.json")
        except:
            pass  # Ignore errors when removing temp files


if __name__ == "__main__":
    main()
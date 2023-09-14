import plotly.graph_objects as go
from PIL import Image

# Load your image using Pillow
image = Image.open("image_0126.png")

# Create a scatter plot (red square) on top of the image
mask_x = [100, 300, 300, 100, 100]
mask_y = [100, 100, 300, 300, 100]

fig = go.Figure()

# Add the image as a background image
fig.add_layout_image(
    source=image,
    x=0,
    y=0,
    xref="x",
    yref="y",
    sizex=image.width,
    sizey=image.height,
    opacity=1,
)

# Create a draggable shape (red square)
fig.add_trace(go.Scatter(x=mask_x, y=mask_y, mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=10, color='red')))
fig.update_shapes(dict(editable=True))

# Set layout properties, e.g., image size and axis options
fig.update_xaxes(range=[0, image.width], showgrid=False, zeroline=False)
fig.update_yaxes(range=[0, image.height], showgrid=False, zeroline=False)

# Set layout properties for interactive features
fig.update_layout(
    dragmode='drawrect',  # Allow drawing rectangles
    showlegend=False,     # Hide legend
    autosize=False,       # Prevent auto-scaling
    width=800,            # Set the width of the plot (adjust as needed)
    height=600,           # Set the height of the plot (adjust as needed)
)

# Show the plot
fig.show()

# Everything is a Graph: Image Segmentation via Graph Cuts

This project implements an image segmentation tool using the Max-Flow Min-Cut algorithm (Edmonds-Karp). It treats an image as a graph where pixels are nodes and edges represent similarity. By finding the minimum cut in the graph, it separates the foreground object from the background.

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)

## Project Overview

While modern image segmentation often relies on Deep Learning, this project explores a classical algorithmic approach. It demonstrates how graph theory can be applied to computer vision tasks, providing interpretability and mathematical robustness.

**Key Features:**
-   **Graph Representation**: Converts pixels into a grid graph.
-   **Intensity-Based Weights**: Edge weights are based on pixel intensity differences.
-   **Max-Flow Min-Cut**: Uses the Edmonds-Karp algorithm to find the optimal separation.
-   **Visualizations**: Displays the original image and the resulting segmentation mask.
-   **Rich Output**: Uses the `rich` library for colorful console output and status updates.

## How It Works

1.  **Preprocessing**: The image is loaded, resized for performance, and converted to grayscale.
2.  **Graph Construction**:
    *   **Nodes**: Each pixel is a node. Two special nodes, Source (S) and Sink (T), are added.
    *   **n-links (Neighbor Links)**: Adjacent pixels are connected. The weight is high if pixels are similar (strong bond) and low if they are different (weak bond).
    *   **t-links (Terminal Links)**:
        *   Pixels at the image border are connected to the Sink (Background seeds).
        *   Pixels at the very center are connected to the Source (Foreground seeds).
3.  **Max-Flow Calculation**: The Edmonds-Karp algorithm finds the maximum flow from Source to Sink. The "bottlenecks" in this flow correspond to the edges that should be cut.
4.  **Segmentation**: The set of nodes reachable from the Source in the residual graph constitutes the foreground object.

## 🔗 Project Structure
```
.
├── images/                    # Directory containing input images and outputs
│   ├── graphics/             # Graphics and visualizations
│   │   └── netflow01.gif
│   ├── jules_tracker/        # Jules tracking related images
│   │   ├── final_stages.png
│   │   ├── jules_request.png
│   │   └── review_plan.png
│   ├── output/               # Generated output images
│   │   ├── test1.png
│   │   ├── test2.png
│   │   ├── test3.png
│   │   └── test4.png
│   └── dragonite_og.jpeg     # Default sample image
├── reports/                   # Project reports and presentations
│   ├── jules-review/
│   │   └── jules_review.ipynb
│   └── presentation/
│       └── everything_is_a_graph.pdf
├── src/                       # Source code directory
│   └── (Python source files)
├── .vscode/                   # VS Code configuration
├── index.html                 # PDF viewer interface
├── style.css                  # Styling for PDF viewer
├── main.py                    # Main Python source code file
├── requirements.txt           # List of Python dependencies
├── .gitignore                 # Git ignore rules
├── LICENSE                    # Project license
└── README.md                  # Project documentation
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the segmentation on the default image:

```bash
python main.py
```

### Customization

You can modify `main.py` to change parameters:

*   **Image Path**: Change the file path in the `ImageGraph` instantiation:
    ```python
    processor = ImageGraph('path/to/your/image.jpg', width=40)
    ```
*   **Resolution**: Adjust the `width` parameter. Higher values give better detail but increase computation time significantly (O(V E^2)).
*   **Seeds**: You can adjust the `seed_radius` or the logic for placing source/sink seeds in the `build_t_links` method to suit different images.

## Author

**Joshua Piña**
<br>Computer Science Department, Georgia State University <br>
*Data Science Senior | Program Manager | U.S Army Veteran*

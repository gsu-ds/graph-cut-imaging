import cv2
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.traceback import install # The Rich Package allows for pretty printing, traceback for error handling and a console to display output
from collections import deque # This will be used for BFS in the Edmonds-Karp algorithm


install()
console = Console()

class ImageGraph: 
    def __init__(self, image_path, width=20):
        self.console = console
        
        # 1. Load and Resize
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Calculate height to maintain aspect ratio
        r = width / self.original_img.shape[1]
        dim = (width, int(self.original_img.shape[0] * r))
        self.img = cv2.resize(self.original_img, dim, interpolation=cv2.INTER_AREA)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        self.h, self.w = self.gray.shape
        self.num_pixels = self.h * self.w
        
        # 2. Graph Setup
        # Node IDs: 0 to num_pixels-1 are pixels. 
        # Last two are Source (S) and Sink (T).
        self.SOURCE = self.num_pixels
        self.SINK = self.num_pixels + 1
        self.total_nodes = self.num_pixels + 2
        
        # Initialize Capacity Matrix (Adjacency Matrix)
        # rows = from_node, cols = to_node
        self.capacity = np.zeros((self.total_nodes, self.total_nodes), dtype=int)

        self._print_status()

    def _print_status(self):
        table = Table(title="Graph Initialization Stats")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Image Dimensions", f"{self.w}x{self.h}")
        table.add_row("Total Pixels", str(self.num_pixels))
        table.add_row("Source Node ID", str(self.SOURCE))
        table.add_row("Sink Node ID", str(self.SINK))
        table.add_row("Matrix Size", f"{self.total_nodes}x{self.total_nodes}")
        
        self.console.print(table)

    def get_node_id(self, row, col):
        return row * self.w + col

    def build_n_links(self):
            """
            Connects adjacent pixels (n-links). 
            TUNED STRATEGY: Thresholding.
            If pixels are similar, make the link nearly unbreakable.
            If pixels are different, make the link weak.
            """
            self.console.print("[bold yellow]Building n-links (Threshold Mode)...[/bold yellow]")
            
            connections = 0
            
            # Sensitivity Threshold
            # Pixels with difference < 30 are considered "The Same Object"
            # Pixels with difference > 30 are considered "Edges"
            DIFF_THRESHOLD = 30 
            
            STRONG_BOND = 1000  # Hard to cut
            WEAK_BOND = 1       # Easy to cut
            
            for r in range(self.h):
                for c in range(self.w):
                    u = self.get_node_id(r, c)
                    
                    neighbors = []
                    if c + 1 < self.w: neighbors.append((r, c+1)) # Right
                    if r + 1 < self.h: neighbors.append((r+1, c)) # Down
                    
                    for nr, nc in neighbors:
                        v = self.get_node_id(nr, nc)
                        
                        val_u = int(self.gray[r, c])
                        val_v = int(self.gray[nr, nc])
                        diff = abs(val_u - val_v)
                        
                        # The Logic Switch
                        if diff < DIFF_THRESHOLD:
                            weight = STRONG_BOND
                        else:
                            weight = WEAK_BOND
                        
                        self.capacity[u, v] = weight
                        self.capacity[v, u] = weight
                        connections += 1

            self.console.print(f"[green]Success![/green] Created {connections} neighbor connections.")
            
    def bfs(self, parent_map):
        """
        Breadth-First Search to find a path from Source to Sink.
        Returns True if path exists, False otherwise.
        """
        visited = [False] * self.total_nodes
        queue = deque()
        
        queue.append(self.SOURCE)
        visited[self.SOURCE] = True
        parent_map[self.SOURCE] = -1
        
        while queue:
            u = queue.popleft()
            
            # If we reached the Sink, we found a path!
            if u == self.SINK:
                return True
            
            # Check all nodes for available capacity
            for v in range(self.total_nodes):
                if not visited[v] and self.capacity[u, v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent_map[v] = u
        return False

    def calculate_max_flow(self):
        """
        The Edmonds-Karp Algorithm.
        Repeatedly finds paths and pushes flow until the graph is full.
        """
        self.console.print("[bold yellow]Running Max-Flow (Edmonds-Karp)...[/bold yellow]")
        
        parent_map = [-1] * self.total_nodes
        max_flow_value = 0
        
        # While a path exists from S -> T
        while self.bfs(parent_map):
            # 1. Find the bottleneck (minimum capacity) along the path
            path_flow = float('inf')
            s = self.SINK
            while(s != self.SOURCE):
                path_flow = min(path_flow, self.capacity[parent_map[s], s])
                s = parent_map[s]
            
            # 2. Update residual capacities
            v = self.SINK
            while(v != self.SOURCE):
                u = parent_map[v]
                self.capacity[u, v] -= path_flow # Reduce forward capacity
                self.capacity[v, u] += path_flow # Add residual (backward) capacity
                v = parent_map[v]
            
            max_flow_value += path_flow

        self.console.print(f"[green]Max Flow Complete![/green] Total Flow: {max_flow_value}")

    def segment_image(self):
        """
        Reconstructs the image based on the 'Min-Cut'.
        Any node still reachable from Source after Max-Flow is the Object.
        """
        self.console.print("[bold yellow]Segmenting Image...[/bold yellow]")
        
        # Final BFS to see what is reachable from Source
        visited = [False] * self.total_nodes
        queue = deque()
        queue.append(self.SOURCE)
        visited[self.SOURCE] = True
        
        while queue:
            u = queue.popleft()
            for v in range(self.total_nodes):
                if not visited[v] and self.capacity[u, v] > 0:
                    queue.append(v)
                    visited[v] = True
        
        # Create the binary mask
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        for r in range(self.h):
            for c in range(self.w):
                node_id = self.get_node_id(r, c)
                if visited[node_id]:
                    mask[r, c] = 1 # Foreground
        
        # Display
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.gray, cmap='gray')
        plt.title("Original Grayscale")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Graph Cut Result")
        plt.axis('off')
        plt.show()    

    def show_capacity_sample(self):
        """Prints a small sample of the capacity matrix for debugging."""
        self.console.print(Panel("Sample of Capacity Matrix (First 5 nodes)", title="Debug"))
        print(self.capacity[:5, :5])

    def show_image(self):
        plt.imshow(self.gray, cmap='gray')
        plt.title(f"Graph Representation ({self.w}x{self.h})")
        plt.axis('off')
        plt.show()

    def build_t_links(self):
            """
            Connects pixels to Source (Foreground) and Sink (Background).
            REFINED STRATEGY: 
            - Only pin the VERY CENTER to Source (Small seed).
            - Pin the edges to Sink.
            - Let the algorithm decide the rest based on color.
            """
            self.console.print("[bold yellow]Building t-links (Terminal connections)...[/bold yellow]")
            
            INF = 1000000000
            source_links = 0
            sink_links = 0
            
            # Define the center point
            mid_r, mid_c = self.h // 2, self.w // 2
            
            # Radius of our "Object Seed" (Make this small! e.g., 2 or 3 pixels)
            seed_radius = 2 
            
            for r in range(self.h):
                for c in range(self.w):
                    u = self.get_node_id(r, c)
                    
                    # 1. Background Seeds (The Image Border)
                    if r == 0 or r == self.h - 1 or c == 0 or c == self.w - 1:
                        self.capacity[u, self.SINK] = INF
                        sink_links += 1
                    
                    # 2. Foreground Seeds (Only a small dot in the center)
                    elif (mid_r - seed_radius <= r <= mid_r + seed_radius) and \
                        (mid_c - seed_radius <= c <= mid_c + seed_radius):
                        self.capacity[self.SOURCE, u] = INF
                        source_links += 1
                        
            self.console.print(f"[green]Success![/green] Linked {source_links} Center seeds and {sink_links} Border seeds.")
# --- EXECUTION ---
if __name__ == "__main__":

    # 1. Initialize (Try width=30 for a better view)
    processor = ImageGraph('Design and Analysis - Algorithms\\project\\data\\images\\dragonite_upscale.jpg', width=40) 
    
    # 2. Build Graph
    processor.build_n_links()
    processor.build_t_links()
    
    # 3. Run Algorithm
    processor.calculate_max_flow()
    
    # 4. View Result
    processor.segment_image()
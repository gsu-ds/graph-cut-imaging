# Everything is a Graph 

Joshua Piña, Computer Science Department, Georgia State University
---
**History:** Data Science Senior | Program Manager | U.S Army Veteran<br>
**Memberships:** FourBlock | ColorStack | CodePath (Prior Student, Current Peer Mentor)<br>
**Cohorts:** Syracuse University O2O (Fall 2025, AWS Solutions Architect)

Viz, Docs, Reports -> [Images are just Graphs (in-progress)](https://google.com)

--- 
### Abstract 
While modern image segmentation increasingly relies on Deep Learning, classical algorithmic approaches remain fundamental for their interpretability and mathematical robustness. Motivated by the challenges of binary segmentation explored in Digital Image Processing, this project implements a solution using the Max-Flow Min-Cut theorem, rather than a neural network.The project models a digital image as a flow network where individual pixels function as nodes in a grid graph. The network is constructed using n-links (neighbor connections) and t-links (terminal connections). Edge capacities between pixel nodes are calculated using an exponential decay function based on pixel intensity differences ($w = e^{-|I_u - I_v|/\sigma}$), which mathematically penalizes "cuts" through uniform regions while incentivizing cuts along high-contrast edges.By implementing the Edmonds-Karp algorithm, the project computes the maximum flow from a defined Source (foreground seed) to a Sink (background seed). The resulting residual graph reveals the Minimum Cut, effectively isolating the object of interest with minimal energy cost. The final application demonstrates successful segmentation of complex, non-convex shapes from grayscale images, validating the efficacy of classical graph algorithms on unstructured visual data.

---
### Breakdown
**Project Concept:** Image Segmentation via Graph Cuts<br>
**Goal:** Implement the Max-Flow Min-Cut algorithm to mathematically separate an object from its background in an image.<br>
**Process:** 
  
  - Use Numpy and OpenCV to convert image to grayscale.
  - Calculate pixel intensity distributions to assign initial weights.
  - Convert image pixels to graph nodes.
  - Connect adjacent pixels and classify them as neighbors with weights based on color similarity.
  - Use the source/sink concept to create links to source node as the foreground and a sink node as the background.
  - Implement Ford-Fulkerson or Edmonds-Karp to find the maximum flow from Source to Sink.
  - Utilize graph cuts at choke points to segment the image.
  - Evaluate and finetune script.
  - (Pending task) Consider converting to RGB for final tests.

---





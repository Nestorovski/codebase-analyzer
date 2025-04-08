# Codebase Analyzer

Codebase Analyzer is a tool to analyze and visualize the structure of any codebase. It generates an interactive graph to explore file relationships and provides detailed analysis metrics.

## Features

- Supports multiple programming languages.
- Handles large codebases.
- Optional AWS infrastructure analysis.
- Generates a JSON representation of the codebase structure.
- Creates an interactive HTML graph for visualization.
- Provides detailed analysis metrics.

## Description

### Overview

This project is a **Codebase Analyzer** that scans a codebase to analyze its structure, relationships, and dependencies. It generates a JSON representation of the codebase, performs advanced analysis, and optionally creates an interactive visualization of the relationships between files, components, and AWS infrastructure.

---

### Key Features

1. **Codebase Scanning**:
   - Recursively scans a directory to identify files, components, and functions.
   - Categorizes files into types (e.g., frontend, backend, utility, test).
   - Extracts dependencies (e.g., imports, references) and builds a graph of relationships.

2. **AWS Infrastructure Analysis**:
   - Parses `serverless.yml` files to detect AWS resources (e.g., Lambda functions, DynamoDB tables, API Gateway endpoints).
   - Maps AWS resources to code components (e.g., Lambda handlers).

3. **Graph Representation**:
   - Builds a graph with nodes (files, components, functions, AWS resources) and edges (relationships like "imports", "invokes").
   - Prevents duplicate nodes using a cache.

4. **Advanced Analysis**:
   - Detects dependency cycles.
   - Identifies potential code duplication (e.g., JSX/TSX pairs).
   - Performs centrality analysis to find the most connected nodes.
   - Groups related nodes into communities using modularity-based clustering.

5. **Visualization**:
   - Creates an interactive graph visualization using Plotly.
   - Nodes are color-coded and sized based on type and connectivity.
   - Relationships are represented as edges with labels (e.g., "imports", "invokes").

6. **Output**:
   - Saves the graph as a JSON file.
   - Optionally generates a Markdown analysis report.
   - Saves the visualization as an HTML file.

---

### How It Works

1. **Configuration**:
   - The script defines file extensions to include, directories/files to exclude, and node/relationship types for categorization.

2. **Codebase Scanning**:
   - The `scan_codebase` function walks through the directory tree, processes each file, and extracts metadata (e.g., size, type, dependencies).
   - It uses regex patterns and AST parsing to extract imports, functions, and React components.

3. **AWS Infrastructure Detection**:
   - The `parse_serverless_config` function parses `serverless.yml` to identify AWS resources and their relationships.
   - The `find_aws_resource_references` function scans code for references to these resources.

4. **Graph Construction**:
   - Nodes and edges are added to the `graph` dictionary using `add_node` and `add_edge` functions.
   - Nodes represent files, components, functions, and AWS resources.
   - Edges represent relationships like "imports", "contains", "invokes".

5. **Analysis**:
   - The `analyze_graph` function performs advanced analysis on the graph using NetworkX.
   - It detects cycles, identifies the most connected nodes, and groups nodes into communities.

6. **Visualization**:
   - The `visualize_graph` function uses Plotly to create an interactive graph.
   - Nodes are grouped by type (e.g., frontend, backend, shared) and displayed with tooltips and color coding.

7. **Output**:
   - The graph is saved as a JSON file.
   - The analysis report is saved as a Markdown file.
   - The visualization is saved as an HTML file.

---

### Files in the Project

#### 1. `requirements.txt`

Specifies the Python dependencies:

- `networkx`: For graph analysis.
- `plotly`: For visualization.
- `pyyaml`: For parsing YAML files (e.g., `serverless.yml`).

#### 2. `README.md`

Provides an overview of the project, its features, and usage instructions.

#### 3. `codebase_analyzer.py`

The main script that implements all the functionality described above.

---

### Important Notes

- **Sensitive Information**: Ensure no sensitive information (e.g., API keys, credentials) is included in the codebase or output files before uploading to GitHub.
- **Licensing**: This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Limitations

- Large codebases may take longer to process.
- Some file types or edge cases may not be fully supported (e.g., uncommon file extensions).

---

### Future Plans

- Add support for additional programming languages.
- Improve performance for large codebases.
- Enhance visualization with more customization options.

---

### Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

---

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Nestorovski/codebase-analyzer.git
    cd codebase-analyzer
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the analyzer:

    ```bash
    python codebase_analyzer.py --root-dir <path_to_codebase>
    ```

    - Replace `<path_to_codebase>` with the path to the directory you want to analyze. If omitted, the script will analyze the current working directory.
    - Example:

        ```bash
        python codebase_analyzer.py --root-dir C:\Users\Username\FolderName
        ```

4. Optional arguments:
    - `--output-dir <path>`: Specify a custom directory for saving output files. Defaults to `<root-dir>/codebasegraphdata`.
    - `--no-visualization`: Skip generating the interactive HTML graph visualization.
    - `--analyze`: Perform deeper analysis of the codebase structure.
    - `--verbose`: Print detailed output during the scan.

5. Outputs:
    - A JSON file representing the codebase graph.
    - An optional Markdown analysis report.
    - An optional interactive HTML graph visualization.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

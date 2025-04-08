#!/usr/bin/env python3
"""
Realestatimator Codebase Analyzer

This script analyzes a codebase to:
1. Generate a JSON representation of file relationships
2. Visualize the relationships in an interactive HTML graph
3. Produce detailed analysis metrics about the codebase structure
4. Detect AWS infrastructure and correlate with code components

Files are saved to a dedicated 'codebasegraphdata' folder with timestamps
to preserve history and avoid conflicts.
"""

import os
import sys
import json
import re
import ast
import datetime
import argparse
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Optional imports with fallback for visualization
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Visualization will be disabled.")
    print("To enable visualization: pip install plotly")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Root directory of the project (default: current script's parent directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directory for saving generated files
OUTPUT_DIR = os.path.join(ROOT_DIR, "codebasegraphdata")

# File extensions to include in analysis
INCLUDE_EXTS = {
    ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".config.js", 
    ".css", ".scss", ".html", ".md", ".py", ".tf", ".tfvars"
}

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    "node_modules", ".git", "build", "dist", "coverage", ".vscode", 
    "__pycache__", "backup", ".next", ".cache", "codebasegraphdata"
}

# Files to exclude from analysis
EXCLUDE_FILES = {
    "package-lock.json", ".DS_Store", "Thumbs.db", ".gitignore", 
    ".env", ".env.local", ".npmrc", ".env.production"
}

# Enhanced node types for better categorization
NODE_TYPES = {
    # Frontend
    "frontend_file": "#4285F4",    # Google Blue
    "component": "#0F9D58",        # Google Green
    "hook": "#DB4437",             # Google Red
    "context": "#F4B400",          # Google Yellow
    "style": "#9C27B0",            # Purple
    
    # Backend
    "backend_file": "#3F51B5",     # Indigo
    "lambda": "#FF5722",           # Deep Orange
    "api-route": "#673AB7",        # Deep Purple
    "aws-resource": "#E91E63",     # Pink
    
    # Shared
    "function": "#ff7f0e",         # Orange
    "service": "#2196F3",          # Light Blue
    "utility": "#607D8B",          # Blue Grey
    "type": "#7f7f7f",             # Gray
    "test": "#17becf",             # Cyan
    "file": "#9E9E9E"              # Gray
}

# Define component categories for better organization
COMPONENT_CATEGORIES = {
    "FRONTEND": ["component", "hook", "context", "style", "frontend_file"],
    "BACKEND": ["lambda", "api-route", "aws-resource", "backend_file"],
    "SHARED": ["function", "service", "utility", "type", "test", "file"]
}

# New relationship types for better edge labeling
RELATIONSHIP_TYPES = {
    "imports": "Imports",
    "contains": "Contains",
    "implements": "Implements",
    "uses": "Uses",
    "invokes": "Invokes",
    "renders": "Renders"
}

# ============================================================================
# GRAPH DATA STRUCTURE
# ============================================================================

# Initialize the graph data structure
graph = {"nodes": [], "edges": []}

# Node cache to prevent duplicates
node_cache = set()

def add_node(node_id, node_type, node_meta=None):
    """Add a node to the graph if it doesn't already exist."""
    if node_id not in node_cache:
        node = {
            "id": node_id,
            "type": node_type,
            "meta": node_meta or {}
        }
        graph["nodes"].append(node)
        node_cache.add(node_id)

def add_edge(from_id, to_id, relationship):
    """Add an edge between two nodes."""
    if from_id != to_id:  # Prevent self-loops
        graph["edges"].append({
            "from": from_id, 
            "to": to_id, 
            "relationship": relationship
        })

# ============================================================================
# AWS INFRASTRUCTURE DETECTION
# ============================================================================

def parse_serverless_config(file_path):
    """Parse serverless.yml to extract AWS resources and their relationships."""
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Remove tabs which yaml doesn't like
        content = content.replace('\t', '  ')
        
        # Parse YAML content
        try:
            serverless_config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {file_path}: {e}")
            return []
        
        if not serverless_config:
            return []
            
        aws_resources = []
        
        # Extract Lambda functions
        if 'functions' in serverless_config:
            for func_name, func_config in serverless_config['functions'].items():
                full_name = f"{serverless_config.get('service', 'service')}-{serverless_config.get('stage', 'dev')}-{func_name}"
                
                # Add the function as a node
                resource_id = f"aws:lambda:{full_name}"
                handler_path = None
                
                if 'handler' in func_config:
                    handler_path = func_config['handler'].split('.')[0]
                    if handler_path.startswith('lambdas/'):
                        handler_path = os.path.join(ROOT_DIR, handler_path)
                    
                add_node(resource_id, "aws-resource", {
                    "name": func_name,
                    "service": "Lambda",
                    "type": "function",
                    "arn": f"arn:aws:lambda:{serverless_config.get('provider', {}).get('region', 'us-east-1')}:{serverless_config.get('accountId', '*')}:function:{full_name}"
                })
                
                aws_resources.append((resource_id, handler_path))
                
                # Look for API Gateway event triggers
                if 'events' in func_config:
                    for event in func_config['events']:
                        if 'http' in event:
                            http_event = event['http']
                            path = http_event.get('path', '')
                            method = http_event.get('method', '').upper()
                            
                            # Add API Gateway endpoint as a node
                            api_id = f"aws:apigateway:{path}:{method}"
                            add_node(api_id, "api-route", {
                                "name": f"{method} {path}",
                                "service": "API Gateway",
                                "method": method,
                                "path": path
                            })
                            
                            # Connect API Gateway to Lambda
                            add_edge(api_id, resource_id, "invokes")
        
        # Extract DynamoDB tables
        if 'resources' in serverless_config and 'Resources' in serverless_config['resources']:
            for resource_name, resource_config in serverless_config['resources']['Resources'].items():
                if resource_config.get('Type') == 'AWS::DynamoDB::Table':
                    table_name = resource_config.get('Properties', {}).get('TableName', resource_name)
                    
                    # Add DynamoDB table as a node
                    resource_id = f"aws:dynamodb:{table_name}"
                    add_node(resource_id, "aws-resource", {
                        "name": table_name,
                        "service": "DynamoDB",
                        "type": "table",
                        "arn": f"arn:aws:dynamodb:{serverless_config.get('provider', {}).get('region', 'us-east-1')}:{serverless_config.get('accountId', '*')}:table/{table_name}"
                    })
                    
                    aws_resources.append((resource_id, None))
        
        return aws_resources
    except Exception as e:
        print(f"Error parsing serverless config {file_path}: {e}")
        return []

def find_aws_resource_references(content, file_path, aws_resources):
    """Find references to AWS resources in code."""
    for resource_id, handler_path in aws_resources:
        # If this is a handler file for a Lambda function, connect it
        if handler_path and os.path.normpath(file_path) == os.path.normpath(f"{handler_path}.js"):
            add_edge(file_path, resource_id, "implements")
            continue
            
        # Check for DynamoDB table references
        if "aws:dynamodb:" in resource_id:
            table_name = resource_id.split(':')[-1]
            # Look for table name in code
            if table_name in content:
                # Basic check - could be improved with regex patterns
                add_edge(file_path, resource_id, "uses")
        
        # Check for Lambda function references
        if "aws:lambda:" in resource_id:
            function_name = resource_id.split(':')[-1]
            # Look for function name in code
            if function_name in content:
                add_edge(file_path, resource_id, "references")

# ============================================================================
# CODE ANALYSIS FUNCTIONS
# ============================================================================

def get_file_type(file_path):
    """Determine the file type based on extension and content with better categorization."""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Determine if this is frontend or backend
    is_frontend = (
        'frontend' in file_path or 
        'src' in file_path or 
        'components' in file_path or 
        'pages' in file_path or
        'hooks' in file_path or
        'context' in file_path
    )
    
    is_backend = (
        'backend' in file_path or 
        'lambda' in file_path or 
        'api' in file_path or
        'aws' in file_path or
        'server' in file_path
    )
    
    # Determine file type with enhanced categorization
    if ext in {".tsx", ".jsx"}:
        return "component"
    elif ext in {".ts", ".js"} and "hook" in file_path.lower():
        return "hook"
    elif ext in {".ts", ".js"} and "service" in file_path.lower():
        return "service"
    elif ext in {".ts", ".js"} and "context" in file_path.lower():
        return "context"
    elif ext in {".ts", ".js"} and "util" in file_path.lower():
        return "utility"
    elif ext in {".d.ts"}:
        return "type"
    elif ext in {".css", ".scss"}:
        return "style"
    elif "test" in file_path.lower() or "__tests__" in file_path:
        return "test"
    elif ext in {".tf", ".tfvars", ".yml", ".yaml"} and ("aws" in file_path.lower() or "serverless" in file_path.lower()):
        return "aws-resource"
    elif "lambda" in file_path.lower():
        return "lambda"
    elif is_frontend:
        return "frontend_file"
    elif is_backend:
        return "backend_file"
    else:
        return "file"

def extract_imports(content, file_path):
    """Extract import statements from code."""
    imports = []
    
    # Different regex patterns for different file types
    if file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
        # ES modules
        import_regex = r'import\s+(?:{[^}]+}|\*\s+as\s+[^\s;]+|[^\s;{]+)\s+from\s+[\'"]([^\'"]+)[\'"]'
        imports.extend(re.findall(import_regex, content))
        
        # CommonJS
        require_regex = r'(?:const|let|var)\s+(?:{[^}]+}|[^\s=]+)\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)'
        imports.extend(re.findall(require_regex, content))
    
    # Clean up imports (remove relative path indicators, etc.)
    cleaned_imports = []
    for imp in imports:
        # Skip npm packages
        if not imp.startswith('.') and not imp.startswith('/'):
            continue
            
        # Normalize path
        cleaned_imports.append(imp)
        
    return cleaned_imports

def find_react_components(content, file_path):
    """Find React component definitions with better component recognition."""
    components = []
    
    if file_path.endswith(('.jsx', '.tsx')):
        # Function components (improved regex)
        component_regex = r'(?:export\s+(?:default\s+)?)?(?:function|const)\s+([A-Z][a-zA-Z0-9]*)\s*(?::|=)'
        components.extend(re.findall(component_regex, content))
        
        # Class components
        class_regex = r'class\s+([A-Z][a-zA-Z0-9]*)\s+extends\s+(?:React\.)?Component'
        components.extend(re.findall(class_regex, content))
        
        # Function components with React.FC typing
        fc_regex = r'(?:export\s+(?:default\s+)?)?(?:const|function)\s+([A-Z][a-zA-Z0-9]*)\s*:\s*(?:React\.)?FC'
        components.extend(re.findall(fc_regex, content))
    
    # Deduplicate components
    return list(set(components))

def find_functions(content, file_path):
    """Find function definitions with improved regex patterns."""
    if file_path.endswith('.py'):
        # Parse Python functions using AST
        try:
            tree = ast.parse(content)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            return []
    elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
        # Enhanced regex patterns for JavaScript/TypeScript functions
        patterns = [
            r'(?:function\s+|const\s+|let\s+|var\s+)([a-zA-Z0-9_]+)\s*(?::|=\s*(?:\([^)]*\)|[^=]*=>))',  # Traditional and const
            r'([a-zA-Z0-9_]+)\s*=>\s*[{(\[]',  # Simple arrow functions
            r'(?:export\s+)?(?:async\s+)?(?:function\s+)?([a-zA-Z0-9_]+)\s*\([^)]*\)\s*{',  # Named functions with params
            r'(?:export\s+const|export\s+function)\s+([a-zA-Z0-9_]+)',  # Exported functions
            r'export\s+default\s+(?:function|const)\s*([a-zA-Z0-9_]+)?',  # Default exports
            r'([a-zA-Z0-9_]+)\s*:\s*function\s*\([^)]*\)',  # Object methods
        ]
        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            functions.extend([match for match in matches if match])
        
        # Handle anonymous default exports
        if re.search(r'export\s+default\s+(?:function|const)\s*(?=\([^)]*\)|\s*{)', content):
            functions.append("anonymous_default_function")
        
        # Remove duplicates and filter out special cases
        return list(set([f for f in functions if f and not f in ['if', 'for', 'while', 'switch', 'return', 'const', 'let', 'var']]))
    
    return []

def extract_dependencies(file_path, base_dir):
    """Extract dependencies from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip binary files
        return []
    
    # Get imported files
    imports = extract_imports(content, file_path)
    dependencies = []
    
    for imp in imports:
        if imp.startswith('.'):
            # Resolve relative import
            source_dir = os.path.dirname(file_path)
            if imp.startswith('./'):
                imp = imp[2:]
            elif imp.startswith('../'):
                source_dir = os.path.dirname(source_dir)
                imp = imp[3:]
            
            # Try different extensions
            for ext in ['.js', '.jsx', '.ts', '.tsx', '.json', '.css', '.scss']:
                potential_path = os.path.normpath(os.path.join(source_dir, imp + ext))
                if os.path.exists(potential_path):
                    dependencies.append(potential_path)
                    break
    
    return dependencies

# ============================================================================
# CODEBASE SCANNING
# ============================================================================

def scan_codebase(root_dir, verbose=False):
    """Scan the codebase and build the graph with improved function tracking."""
    if verbose:
        print(f"Scanning codebase at: {root_dir}")
    
    # Track file statistics
    file_stats = {
        "total_files": 0,
        "by_type": defaultdict(int),
        "by_extension": defaultdict(int),
        "largest_files": []
    }
    
    # Look for serverless.yml to extract AWS resources
    serverless_path = os.path.join(root_dir, "serverless.yml")
    aws_resources = []
    if os.path.exists(serverless_path):
        if verbose:
            print(f"Found serverless.yml, extracting AWS resources...")
        aws_resources = parse_serverless_config(serverless_path)
        if verbose:
            print(f"Extracted {len(aws_resources)} AWS resources")
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        for filename in files:
            if filename in EXCLUDE_FILES or filename.startswith('.'):
                continue
                
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in INCLUDE_EXTS:
                continue
                
            # Get file metadata
            try:
                file_size = os.path.getsize(file_path)
                file_stats["total_files"] += 1
                file_stats["by_extension"][file_ext] += 1
            except OSError:
                continue
                
            if verbose:
                rel_path = os.path.relpath(file_path, root_dir)
                print(f"Processing: {rel_path}")
            
            # Get file type
            file_type = get_file_type(file_path)
            file_stats["by_type"][file_type] += 1
            
            # Track largest files
            file_stats["largest_files"].append((file_path, file_size))
            if len(file_stats["largest_files"]) > 100:
                file_stats["largest_files"].sort(key=lambda x: x[1], reverse=True)
                file_stats["largest_files"] = file_stats["largest_files"][:100]
            
            # Add file node
            rel_path = os.path.relpath(file_path, root_dir)
            add_node(file_path, file_type, {
                "name": filename,
                "path": rel_path,
                "size": file_size,
                "extension": file_ext,
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            })
            
            # Extract code content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract components (for React files)
                components = find_react_components(content, file_path)
                for comp in components:
                    comp_id = f"{file_path}::{comp}"
                    add_node(comp_id, "component", {"name": comp})
                    add_edge(file_path, comp_id, "contains")
                
                # Extract functions with improved tracking
                functions = find_functions(content, file_path)
                if verbose and functions:
                    print(f"Found {len(functions)} functions in {rel_path}: {functions}")
                for func in functions:
                    func_id = f"{file_path}::{func}"
                    add_node(func_id, "function", {"name": func})
                    add_edge(file_path, func_id, "contains")
                    file_stats["by_type"]["function"] += 1
                
                # Extract dependencies
                dependencies = extract_dependencies(file_path, root_dir)
                for dep in dependencies:
                    if os.path.exists(dep):
                        add_edge(file_path, dep, "imports")
                
                # Find AWS resource references
                if aws_resources:
                    find_aws_resource_references(content, file_path, aws_resources)
                    
            except Exception as e:
                if verbose:
                    print(f"Error processing {file_path}: {e}")
    
    # Sort largest files
    file_stats["largest_files"].sort(key=lambda x: x[1], reverse=True)
    file_stats["largest_files"] = [(os.path.relpath(p, root_dir), s) for p, s in file_stats["largest_files"][:20]]
    
    return file_stats

# ============================================================================
# ADVANCED GRAPH ANALYSIS
# ============================================================================

def analyze_graph(G, file_stats, output_dir, timestamp):
    """Perform deep analysis on the graph structure and export results."""
    analysis_results = []
    
    # Basic stats
    analysis_results.append("# Codebase Analysis Report\n")
    analysis_results.append(f"**Generated at:** {datetime.datetime.now().isoformat()}\n")
    analysis_results.append(f"**Total Nodes:** {G.number_of_nodes()}\n")
    analysis_results.append(f"**Total Edges:** {G.number_of_edges()}\n")
    
    # Node type breakdown
    node_types = defaultdict(int)
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        node_types[node_type] += 1
    
    analysis_results.append("\n## Node Types Breakdown\n")
    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        analysis_results.append(f"- **{node_type}**: {count}\n")
    
    # Detect potential code duplication (JSX/TSX pairs)
    file_nodes = [n for n in G.nodes() if G.nodes[n]['type'] in ('file', 'frontend_file', 'backend_file')]
    jsx_files = [n for n in file_nodes if n.endswith('.jsx')]
    tsx_files = [n for n in file_nodes if n.endswith('.tsx')]
    
    potential_duplicates = []
    for jsx in jsx_files:
        jsx_base = os.path.basename(jsx).replace('.jsx', '')
        for tsx in tsx_files:
            tsx_base = os.path.basename(tsx).replace('.tsx', '')
            if jsx_base == tsx_base:
                potential_duplicates.append((jsx, tsx))
    
    if potential_duplicates:
        analysis_results.append("\n## Potential Code Duplication (JSX/TSX Pairs)\n")
        for jsx, tsx in potential_duplicates:
            analysis_results.append(f"- {os.path.relpath(jsx, ROOT_DIR)} ↔ {os.path.relpath(tsx, ROOT_DIR)}\n")
    
    # Detect dependency cycles
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            analysis_results.append("\n## Dependency Cycles Detected\n")
            for cycle in cycles[:5]:  # Limit to first 5 for brevity
                cycle_path = " -> ".join(os.path.relpath(node, ROOT_DIR) if isinstance(node, str) else str(node) for node in cycle)
                analysis_results.append(f"- Cycle: {cycle_path}\n")
    except Exception as e:
        analysis_results.append(f"\n## Dependency Cycle Detection Error\n- {str(e)}\n")
    
    # Centrality analysis (most connected nodes)
    degree_centrality = nx.degree_centrality(G)
    most_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    
    analysis_results.append("\n## Most Connected Nodes (Degree Centrality)\n")
    for node, centrality in most_connected:
        node_type = G.nodes[node].get('type', 'unknown')
        rel_path = os.path.relpath(node, ROOT_DIR) if isinstance(node, str) and os.path.isfile(node) else str(node)
        analysis_results.append(f"- **{rel_path}** ({node_type}): Centrality = {centrality:.3f}\n")
    
    # Community detection for grouping related nodes
    try:
        G_undir = G.to_undirected()
        communities = nx.algorithms.community.greedy_modularity_communities(G_undir)
        analysis_results.append(f"\n## Community Detection\nDetected {len(communities)} communities:\n")
        for i, community in enumerate(sorted(communities, key=len, reverse=True)[:3]):
            files = [n for n in community if isinstance(n, str) and os.path.isfile(n)]
            analysis_results.append(f"- **Community {i+1}**: {len(community)} nodes, {len(files)} files\n")
            for file in files[:3]:
                analysis_results.append(f"  - {os.path.relpath(file, ROOT_DIR)}\n")
    except Exception as e:
        analysis_results.append(f"\n## Community Detection Error\n- {str(e)}\n")
    
    # Export analysis to Markdown
    analysis_path = os.path.join(output_dir, f"analysis_report_{timestamp}.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(analysis_results))
    print(f"Analysis report saved to: {analysis_path}")

# ============================================================================
# NETWORK GRAPH CREATION & VISUALIZATION
# ============================================================================

def create_network_graph(graph_data):
    """Create a NetworkX graph from the graph data."""
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in graph_data["nodes"]:
        G.add_node(node["id"], 
                  type=node["type"], 
                  meta=node["meta"])
    
    # Add edges
    for edge in graph_data["edges"]:
        G.add_edge(edge["from"], edge["to"], 
                  relationship=edge["relationship"])
    
    return G

def get_node_category(node_type):
    """Determine the category of a node based on its type."""
    for category, types in COMPONENT_CATEGORIES.items():
        if node_type in types:
            return category
    return "SHARED"

def visualize_graph(G, file_stats, timestamp):
    """Create an interactive visualization of the graph using Plotly with enhanced layout."""
    if not PLOTLY_AVAILABLE:
        print("Plotly is not available. Skipping visualization.")
        return None
    
    # Dynamic repulsion based on node count for better separation
    k = 1.5 + (len(G.nodes()) / 400)  # Increased base repulsion
    iterations = min(300, max(150, len(G.nodes()) // 2))  # More iterations for convergence
    
    # Group nodes by category
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'file')
        category = get_node_category(node_type)
        G.nodes[node]['layer'] = {'FRONTEND': 0, 'BACKEND': 1, 'SHARED': 2}.get(category, 2)
    
    try:
        # Initial positions with multipartite layout for clearer category separation
        initial_pos = nx.multipartite_layout(G, subset_key='layer', scale=2.5)  # Increased scale
        # Refine with spring layout
        pos = nx.spring_layout(G, pos=initial_pos, k=k, iterations=iterations, seed=42)
    except Exception:
        print("Error computing graph layout. Falling back to circular layout.")
        pos = nx.circular_layout(G)
    
    # Create node traces by category for better visual grouping
    node_traces = {}
    node_counts = defaultdict(int)
    
    # Compute counts directly from the graph
    function_count = sum(1 for node in G.nodes() if G.nodes[node].get('type') == 'function')
    component_count = sum(1 for node in G.nodes() if G.nodes[node].get('type') == 'component')
    aws_resource_count = sum(1 for node in G.nodes() if G.nodes[node].get('type') in ('aws-resource', 'lambda'))
    
    # Prepare the data with improved visibility
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'file')
        node_counts[node_type] += 1
        
        # Create a trace for this node type if it doesn't exist
        if node_type not in node_traces:
            node_traces[node_type] = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                name=node_type.replace('_', ' ').title(),
                marker=dict(
                    size=[],
                    color=NODE_TYPES.get(node_type, "#7f7f7f"),
                    line=dict(width=1, color='#ffffff')
                )
            )
        
        meta = G.nodes[node].get('meta', {})
        x, y = pos[node]
        
        node_traces[node_type]['x'] = node_traces[node_type]['x'] + (x,)
        node_traces[node_type]['y'] = node_traces[node_type]['y'] + (y,)
        
        # Enhanced node size based on importance and connectivity
        size_factor = 1.0
        if node_type == "component" or node_type == "lambda" or node_type == "service":
            size_factor = 1.8
        elif node_type == "function":
            size_factor = 1.5
            
        connections = len(list(G.successors(node))) + len(list(G.predecessors(node)))
        size = 8 + (connections * 0.6 * size_factor)  # Adjusted base size
        node_traces[node_type]['marker']['size'] = node_traces[node_type]['marker']['size'] + (size,)
        
        # Enhanced tooltip with more detailed information
        name = meta.get('name', os.path.basename(node) if isinstance(node, str) else str(node))
        path = meta.get('path', '')
        if not path and isinstance(node, str) and os.path.isfile(node):
            path = os.path.relpath(node, ROOT_DIR)
        
        # Build a detailed tooltip with HTML formatting
        tooltip = f"<b>{name}</b><br><i>{node_type.replace('_', ' ').title()}</i>"
        if path:
            tooltip += f"<br>Path: {path}"
        
        # Add connection count to tooltip
        incoming = len(list(G.predecessors(node)))
        outgoing = len(list(G.successors(node)))
        tooltip += f"<br>Connections: {connections} ({incoming} in, {outgoing} out)"
        
        node_traces[node_type]['text'] = node_traces[node_type]['text'] + (tooltip,)
    
    # Create edge traces with relationship labels and reduced opacity
    edge_traces = {}
    
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        relationship = edge[2].get('relationship', 'connects')
        
        # Create a trace for this relationship if it doesn't exist
        if relationship not in edge_traces:
            edge_traces[relationship] = go.Scatter(
                x=[],
                y=[],
                line=dict(
                    width=1.5 if relationship in ['invokes', 'implements'] else 1,
                    color={
                        'imports': 'rgba(33, 150, 243, 0.3)',
                        'contains': 'rgba(76, 175, 80, 0.2)',
                        'implements': 'rgba(255, 152, 0, 0.4)',
                        'uses': 'rgba(156, 39, 176, 0.3)',
                        'invokes': 'rgba(244, 67, 54, 0.4)',
                        'renders': 'rgba(96, 125, 139, 0.3)',
                    }.get(relationship, 'rgba(136, 136, 136, 0.2)'),
                    shape='spline'  # Use spline for curved edges
                ),
                hoverinfo='text',
                mode='lines',
                name=RELATIONSHIP_TYPES.get(relationship, relationship.title()),
                text=[]
            )
        
        source_meta = G.nodes[source].get('meta', {})
        target_meta = G.nodes[target].get('meta', {})
        
        source_name = source_meta.get('name', os.path.basename(source) if isinstance(source, str) else str(source))
        target_name = target_meta.get('name', os.path.basename(target) if isinstance(target, str) else str(target))
        
        # Edge hover text
        relationship_label = RELATIONSHIP_TYPES.get(relationship, relationship.title())
        edge_tooltip = f"{source_name} {relationship_label} {target_name}"
        
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        edge_traces[relationship]['x'] += (x0, x1, None)
        edge_traces[relationship]['y'] += (y0, y1, None)
        edge_traces[relationship]['text'] += (edge_tooltip, edge_tooltip, None)
    
    # Combine all traces
    data = list(edge_traces.values()) + list(node_traces.values())
    
    # Create figure with better layout and annotations
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='<br>Realestatimator Codebase Visualization',
            titlefont=dict(size=20),
            showlegend=True,
            legend=dict(
                title="Component Types",
                groupclick="toggleitem"
            ),
            hovermode='closest',
            margin=dict(b=40, l=5, r=5, t=70),
            annotations=[
                dict(
                    text=f"Files: {file_stats['total_files']} | " + 
                         f"Components: {component_count} | " +
                         f"Functions: {function_count} | " +
                         f"AWS Resources: {aws_resource_count}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.01,
                    font=dict(size=14)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{"visible": [True] * len(data)}, {"xaxis.range": None, "yaxis.range": None}],
                            label="All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [
                                      t < len(edge_traces) or
                                      (t >= len(edge_traces) and list(node_traces.keys())[t - len(edge_traces)] in 
                                       COMPONENT_CATEGORIES["FRONTEND"] + ["function"])
                                      for t in range(len(data))
                                  ]}, 
                                  {"xaxis.range": None, "yaxis.range": None}],
                            label="Frontend Only",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [
                                      t < len(edge_traces) or
                                      (t >= len(edge_traces) and list(node_traces.keys())[t - len(edge_traces)] in 
                                       COMPONENT_CATEGORIES["BACKEND"] + ["function"])
                                      for t in range(len(data))
                                  ]},
                                  {"xaxis.range": None, "yaxis.range": None}],
                            label="Backend Only",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [
                                      t < len(edge_traces) or
                                      (t >= len(edge_traces) and list(node_traces.keys())[t - len(edge_traces)] == "function")
                                      for t in range(len(data))
                                  ]},
                                  {"xaxis.range": None, "yaxis.range": None}],
                            label="Functions Only",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [
                                      t < len(edge_traces) or
                                      (t >= len(edge_traces) and list(node_traces.keys())[t - len(edge_traces)] == "component")
                                      for t in range(len(data))
                                  ]},
                                  {"xaxis.range": None, "yaxis.range": None}],
                            label="Components Only",
                            method="update"
                        )
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
    )
    
    return fig

# ============================================================================
# MAIN FUNCTION AND COMMAND LINE INTERFACE
# ============================================================================

def main(args):
    """Main entry point for the script."""
    # Use the provided root directory or default to the current working directory
    root_dir = args.root_dir or os.getcwd()
    output_dir = args.output_dir or os.path.join(root_dir, "codebasegraphdata")
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"Analyzing codebase at: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")

    # Scan the codebase and build the graph
    start_time = datetime.datetime.now()
    file_stats = scan_codebase(root_dir, args.verbose)
    scan_time = datetime.datetime.now() - start_time

    # Save the graph data to a JSON file
    json_filename = f"codebase_graph_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    # Create metadata for the report
    metadata = {
        "timestamp": timestamp,
        "generated_at": datetime.datetime.now().isoformat(),
        "root_directory": root_dir,
        "file_count": file_stats["total_files"],
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "scan_time_seconds": scan_time.total_seconds(),
        "aws_resources_detected": len([n for n in graph["nodes"] if n.get("type") == "aws-resource"]),
        "api_routes_detected": len([n for n in graph["nodes"] if n.get("type") == "api-route"]),
        "generator_version": "1.2.0",
        "generator_command": " ".join(sys.argv)
    }

    # Obscure any credentials or sensitive data in the graph
    safe_graph = {"nodes": [], "edges": graph["edges"]}

    for node in graph["nodes"]:
        safe_node = node.copy()
        if node["type"] == "file":
            filepath = node["id"]
            filename = os.path.basename(filepath).lower()
            if any(term in filename for term in ['key', 'secret', 'password', 'credential', '.env', 'token']):
                if "content" in safe_node.get("meta", {}):
                    safe_node["meta"]["content"] = "[REDACTED FOR SECURITY]"
        safe_graph["nodes"].append(safe_node)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "graph": safe_graph,
            "stats": file_stats
        }, f, indent=2)

    print(f"Graph data saved to: {json_path}")

    # Create NetworkX graph
    G = create_network_graph(graph)

    # Perform advanced analysis if requested
    if args.analyze:
        analyze_graph(G, file_stats, output_dir, timestamp)

    # Visualize the graph if not disabled
    if not args.no_visualization and PLOTLY_AVAILABLE:
        print("Generating visualization...")
        fig = visualize_graph(G, file_stats, timestamp)

        if fig:
            html_filename = f"codebase_visualization_{timestamp}.html"
            html_path = os.path.join(output_dir, html_filename)
            fig.write_html(html_path)
            print(f"Visualization saved to: {html_path}")

    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("Note: Add this directory to your .gitignore file to avoid committing potentially sensitive information.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and visualize codebase structure.')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Root directory of the codebase to analyze (default: current working directory)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory where output files will be saved (default: <root-dir>/codebasegraphdata)')
    parser.add_argument('--no-visualization', action='store_true',
                        help='Skip generating visualization (useful for headless environments)')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform deeper analysis of the codebase structure')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output during scanning')
    parser.add_argument('--include-aws', action='store_true',
                        help='Include AWS infrastructure analysis (requires yaml package)')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

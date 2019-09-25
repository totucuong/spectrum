def print_trace(trace):
    for name, node in trace.nodes.items():
        if node['type'] == 'sample':
            print(f'{node["name"]} - sampled value {node["value"]}')

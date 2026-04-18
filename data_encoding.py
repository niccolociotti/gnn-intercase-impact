import networkx as nx
from ast import literal_eval
from os.path import join
from os import makedirs
import json
from numpy import array, random
from pm4py import get_end_activities
from pm4py.objects.conversion.log import converter as log_converter
from pandas import to_datetime, read_csv
from pm4py.statistics.start_activities.log.get import get_start_activities
from torch_geometric.data import Data
import torch
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config import INPUT_PATH, MAX_WORKERS
_shared = None
random.seed(0)


# nodo fittizio collegato al primo evento del prefisso attivo più gli eventi concorrenti
def _var1(active_ig):
    global _shared
    name, igs, ohe, features, path = (
        _shared['name'],
        _shared['igs'],
        _shared['ohe'],
        _shared['features'],
        _shared['path']
    )
    fict_id = 'fict_node|fict_activity|'
    fict_data = {'activity': 'fict_activity'}
    fict_data.update({f: 0.0 for f in features})

    prefix = Data()
    case_nodes = [(n, d) for n, d in active_ig.nodes(data=True)]
    nodes, edges, visited = list(), list(), list()

    i = 0
    while i < len(case_nodes)-2:

        # add active node
        node_id, node_data = case_nodes[i]
        nodes.append([node_id, node_data])
        visited.append(node_id)

        # add active edges
        edges_to_add = [(p, node_id) for p in list(active_ig.predecessors(node_id)) if p in visited]
        edges_to_add += [(node_id, s) for s in list(active_ig.successors(node_id)) if s in visited]
        edges.extend(edges_to_add)

        # dummy node
        if i == 0:
            nodes.append([fict_id, fict_data])
            visited.append(fict_id)
            edges.append((fict_id, node_id))

        # encode active prefix
        node_ids, node_features = zip(*nodes)
        features_x = [ohe[node['activity']].copy() + [node[f] for f in features] for node in node_features]
        edge_index = [[node_ids.index(src), node_ids.index(dst)] for src, dst in edges]

        # concurrent nodes
        concurrent_nodes = literal_eval(node_data['concurrent_selected'])
        if len(concurrent_nodes):
            for concurrent_node in concurrent_nodes:
                features_x.extend([concurrent_node])
                edge_index.extend([[node_ids.index(fict_id), len(features_x)-1]])

        prefix.x = torch.tensor(features_x, dtype=torch.float)
        prefix.edge_index = torch.tensor(edge_index, dtype=torch.int64).T
        prefix.next_activity = case_nodes[i+1][1]['activity']
        prefix.y = torch.tensor([ohe[prefix.next_activity]], dtype=torch.float)
        prefix.active_prefix_size, prefix.case_id, prefix.set = i+1, active_ig.case_id, active_ig.set
        prefix.concurrent_nodes = node_data['n_concurrent_nodes']
        if i >= 1:
            torch.save(prefix, join(path, f'{name}_{prefix.case_id}_{i+1}.pt'))
        i += 1


def generate_prefix_igs(igs, variant, shared_items):
    #if variant == 'var1':
    if variant == 'var_fict_200K_2':
        f = partial(_var1)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(shared_items,)) as executor:
        for _ in tqdm(executor.map(f, igs), total=len(igs), desc="Processing prefix-IGs"):
            pass

    with open(join(shared_items['path'], 'done'), 'w') as _:
        pass


def _discovery(case_id):
    global _shared
    name, cases_time, nodes, igs, activities_to_filter, ohe, k, features = (
        _shared['name'],
        _shared['cases_time'],
        _shared['nodes'],
        _shared['igs'],
        _shared['activities_to_filter'],
        _shared['ohe'],
        _shared['k'],
        _shared['features']
    )

    ig = igs.loc[igs['case_id'] == case_id]
    g = nx.DiGraph()
    g.case_id, g.set = case_id, ig['set'].drop_duplicates().tolist()[0]

    for event in ig.itertuples(index=False):
        if event.type == 'v':
            n_concurrent_nodes, selected_nodes = 0, []
            if k >= 1:
                # get concurrent cases and remove itself
                start_time = to_datetime(event.start_time)
                active_cases = set(cases_time.loc[
                                       (start_time <= cases_time['max']) &
                                       (case_id != cases_time['case_id']), 'case_id'].tolist())

                # keep concurrent events, i.e., events that run when x runs, except end activities
                events = nodes.loc[(
                        (nodes['case_id'].isin(active_cases)) &
                        (~nodes['activity'].isin(activities_to_filter)) &
                        (nodes['start_time'] <= start_time) & (start_time <= nodes['end_time'])
                )]

                concurrent_nodes = array([
                    ohe[act] + [norm, trace, prev] for act, norm, trace, prev in zip(
                        events['activity'], events['norm_time'],
                        events['trace_time'], events['prev_event_time']
                    )], dtype=float)

                n_concurrent_nodes = concurrent_nodes.shape[0]
                if n_concurrent_nodes:
                    # dimezziamo il contributo degli eventi concorrenti
                    concurrent_nodes /= 2
                    if n_concurrent_nodes > k:
                        random.shuffle(concurrent_nodes)
                        selected_nodes = concurrent_nodes[:k].tolist()
                    else:
                        selected_nodes = concurrent_nodes.tolist()

            g.add_node(event.node1,
                       activity=event.activity,
                       norm_time=event.norm_time,
                       trace_time=event.trace_time,
                       prev_event_time=event.prev_event_time,
                       n_concurrent_nodes=n_concurrent_nodes,
                       concurrent_selected=f'{selected_nodes}'
                       )

        elif event.type == 'e':
            g.add_edge(event.node1, event.node2)

    return g if verify_g(g) else None


def discovery_inter_cases(cases, shared_items):
    f = partial(_discovery)
    result = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(shared_items,)) as executor:
        for out in tqdm(executor.map(f, cases), total=len(cases), desc="Discovering inter cases"):
            if out is not None:
                result.append(out)
    return result


def get_ohe_encoding(name, items, attribute):
    ohe = {item:[1 if i == j else 0 for j in range(len(items))] for i, item in enumerate(items)}
    with open(join('dataset', f'{name}_ohe_{attribute}.json'), 'w') as f:
        f.write(json.dumps(ohe, indent=1))
    return ohe


def verify_g(ig):
    # disconnected graph
    if len(list(nx.connected_components(ig.to_undirected()))) != 1:
        print(f'Skipping disconnected prefix: {ig.case_id}, size: {len(ig.nodes)}')
        return False

    return True


def init_worker(shared_data):
    global _shared
    _shared = shared_data


def main(name, variant, k):
    igs = read_csv(join(INPUT_PATH, f'{name}_processed.g'), header=0, sep=',')
    nodes = igs.loc[igs['type'] == 'v']
    case_ids = nodes['case_id'].drop_duplicates().tolist()

    activities_to_filter = ['artificial_start', 'artificial_end']
    event_log = nodes[['case_id', 'activity', 'start_time']].copy()
    event_log = event_log[~event_log['activity'].isin(activities_to_filter)]
    event_log.rename(columns={'case_id': 'case:concept:name', 'activity': 'concept:name', 'start_time': 'time:timestamp'}, inplace=True)

    event_log = log_converter.apply(event_log, variant=log_converter.Variants.TO_EVENT_LOG)
    end_activities = get_end_activities(event_log)
    activities_to_filter += [act for act, freq in end_activities.items()]
    del event_log

    nodes['start_time'] = to_datetime(nodes['start_time'], format='ISO8601', utc=True)
    nodes['end_time'] = to_datetime(nodes['end_time'], format='ISO8601', utc=True)
    cases_time = nodes.groupby("case_id")["end_time"].agg(["min", "max"]).reset_index()

    ohe = get_ohe_encoding(name, nodes['activity'].drop_duplicates().tolist(), 'activities')
    features = ['norm_time', 'trace_time', 'prev_event_time']

    shared_items = {
        'name': name,
        'igs': igs,
        'k': k,
        'activities_to_filter': activities_to_filter,
        'cases_time': cases_time,
        'nodes': nodes,
        'ohe': ohe,
        'features': features
    }
    igs = discovery_inter_cases(case_ids, shared_items)

    ohe['fict_activity'] = len(ohe)*[0]
    path = join('dataset', f'{variant}_{name}_{k}_k_tensors')
    makedirs(path, exist_ok=True)
    
    shared_items.update({
        'igs': igs,
        'path': path,
        'k': k,
        'ohe': ohe
    })
    generate_prefix_igs(igs, variant, shared_items)
    del shared_items, igs, nodes
    gc.collect()


"""
if __name__ == '__main__':
    main('Helpdesk_no_resources', 'var3', 0)
"""

import torch

from beater.sr_memory import EntityGraph


def test_graph_prunes_old_nodes():
    graph = EntityGraph(max_nodes=3)
    for idx in range(4):
        graph.add_entity("test", torch.randn(16), {"idx": idx}, node_id=f"n{idx}")
    assert len(graph.nodes) == 3
    assert "n0" not in graph.nodes


def test_assoc_returns_similar_node():
    graph = EntityGraph(max_nodes=5)
    anchor = torch.ones(8)
    nid = graph.add_entity("test", anchor)
    query = anchor + 0.01 * torch.randn_like(anchor)
    result = graph.assoc(query, top_k=1)
    assert result and result[0][0] == nid

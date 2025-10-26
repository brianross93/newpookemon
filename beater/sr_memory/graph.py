"""Symbolic entity graph supporting ASSOC/FOLLOW/WRITE operations."""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass(slots=True)
class EntityNode:
    node_id: str
    kind: str
    embedding: torch.Tensor
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RelationEdge:
    edge_id: str
    src: str
    dst: str
    rel_type: str
    attrs: Dict[str, object] = field(default_factory=dict)


class EntityGraph:
    """Lightweight adjacency map with cosine-similarity ASSOC."""

    def __init__(self, max_nodes: int = 4096, device: Optional[torch.device] = None):
        self.nodes: Dict[str, EntityNode] = {}
        self.edges: Dict[str, RelationEdge] = {}
        self._adj_out: Dict[str, List[str]] = {}
        self._adj_in: Dict[str, List[str]] = {}
        self.max_nodes = max_nodes
        self.device = device or torch.device("cpu")

    # ----------------------------------------------------------------- mutations
    def add_entity(
        self,
        kind: str,
        embedding: torch.Tensor,
        attrs: Optional[Dict[str, object]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or str(uuid.uuid4())
        self.nodes[nid] = EntityNode(
            node_id=nid,
            kind=kind,
            embedding=embedding.detach().to(self.device),
            attrs=attrs or {},
        )
        self._adj_out.setdefault(nid, [])
        self._adj_in.setdefault(nid, [])
        self._prune_if_needed()
        return nid

    def add_relation(
        self,
        src: str,
        dst: str,
        rel_type: str,
        attrs: Optional[Dict[str, object]] = None,
        edge_id: Optional[str] = None,
    ) -> str:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist before adding relation")
        eid = edge_id or str(uuid.uuid4())
        self.edges[eid] = RelationEdge(
            edge_id=eid, src=src, dst=dst, rel_type=rel_type, attrs=attrs or {}
        )
        self._adj_out[src].append(eid)
        self._adj_in[dst].append(eid)
        return eid

    def write_entity(self, node_id: str, embedding: torch.Tensor, attrs: Dict[str, object]) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} missing")
        node.embedding = embedding.detach().to(self.device)
        node.attrs.update(attrs)

    # ------------------------------------------------------------------- queries
    def assoc(
        self,
        query: torch.Tensor,
        top_k: int = 1,
        filter_kind: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Return node ids and cosine similarities."""
        if not self.nodes:
            return []
        query = query.to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q_norm = torch.norm(query, dim=-1, keepdim=True).clamp(min=1e-6)
        query = query / q_norm

        heap: List[Tuple[float, str]] = []
        for node_id, node in self.nodes.items():
            if filter_kind and node.kind != filter_kind:
                continue
            node_emb = node.embedding.unsqueeze(0)
            sim = torch.cosine_similarity(query, node_emb, dim=-1)[0]
            heapq.heappush(heap, (float(sim), node_id))
            if len(heap) > top_k:
                heapq.heappop(heap)
        # Highest similarity last in heap; sort descending.
        ranked = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(nid, score) for score, nid in ranked]

    def follow(self, node_id: str, rel_type: Optional[str] = None) -> List[str]:
        eids = self._adj_out.get(node_id, [])
        dsts = []
        for eid in eids:
            edge = self.edges[eid]
            if rel_type and edge.rel_type != rel_type:
                continue
            dsts.append(edge.dst)
        return dsts

    def incoming(self, node_id: str, rel_type: Optional[str] = None) -> List[str]:
        eids = self._adj_in.get(node_id, [])
        srcs = []
        for eid in eids:
            edge = self.edges[eid]
            if rel_type and edge.rel_type != rel_type:
                continue
            srcs.append(edge.src)
        return srcs

    # ---------------------------------------------------------------- housekeeping
    def _prune_if_needed(self) -> None:
        if len(self.nodes) <= self.max_nodes:
            return
        # Drop oldest nodes (FIFO) while keeping adjacency consistent.
        overflow = len(self.nodes) - self.max_nodes
        for node_id in list(self.nodes.keys())[:overflow]:
            self._remove_node(node_id)

    def _remove_node(self, node_id: str) -> None:
        for eid in list(self._adj_out.get(node_id, [])):
            self._remove_edge(eid)
        for eid in list(self._adj_in.get(node_id, [])):
            self._remove_edge(eid)
        self.nodes.pop(node_id, None)
        self._adj_out.pop(node_id, None)
        self._adj_in.pop(node_id, None)

    def _remove_edge(self, edge_id: str) -> None:
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return
        if edge_id in self._adj_out.get(edge.src, []):
            self._adj_out[edge.src].remove(edge_id)
        if edge_id in self._adj_in.get(edge.dst, []):
            self._adj_in[edge.dst].remove(edge_id)

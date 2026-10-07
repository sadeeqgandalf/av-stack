"""Mission-level routing: A* over a lane graph.

The road is discretised into (lane, segment) nodes. Edges go forward to the
same lane (cost = segment length) or diagonally to an adjacent lane (extra
lane-change cost). Closed lanes remove nodes. The resulting route tells the
behaviour layer which lane it *should* be in at every point along the road.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from avstack.map.hd_map import HDMap


@dataclass(frozen=True, order=True)
class RouteNode:
    lane: int
    seg: int


class Router:
    def __init__(self, hd_map: HDMap, segment_length: float = 25.0, lane_change_cost: float = 30.0):
        self.map = hd_map
        self.seg_len = segment_length
        self.lc_cost = lane_change_cost
        self.num_segs = int(np.ceil(hd_map.reference.length / segment_length))

    def _blocked(self, node: RouteNode) -> bool:
        s0 = node.seg * self.seg_len
        return any(self.map.is_closed(node.lane, s) for s in np.linspace(s0, s0 + self.seg_len, 5))

    def _neighbors(self, n: RouteNode):
        if n.seg + 1 >= self.num_segs:
            return
        for dl in (-1, 0, 1):
            lane = n.lane + dl
            if not 0 <= lane < self.map.num_lanes:
                continue
            nxt = RouteNode(lane, n.seg + 1)
            if self._blocked(nxt):
                continue
            yield nxt, self.seg_len + (self.lc_cost if dl else 0.0)

    def plan(self, start_s: float, start_lane: int, goal_s: float, goal_lane: int | None = None
             ) -> list[RouteNode]:
        """A* from (start_lane, start_s) to goal_s (optionally in goal_lane)."""
        start = RouteNode(start_lane, int(start_s // self.seg_len))
        goal_seg = min(int(goal_s // self.seg_len), self.num_segs - 1)

        def h(n: RouteNode) -> float:
            return (goal_seg - n.seg) * self.seg_len

        def is_goal(n: RouteNode) -> bool:
            return n.seg == goal_seg and (goal_lane is None or n.lane == goal_lane)

        open_set = [(h(start), 0.0, start)]
        came_from: dict[RouteNode, RouteNode] = {}
        g = {start: 0.0}
        while open_set:
            _, cost, cur = heapq.heappop(open_set)
            if cost > g.get(cur, np.inf):
                continue
            if is_goal(cur):
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return path[::-1]
            for nxt, c in self._neighbors(cur):
                ng = cost + c
                if ng < g.get(nxt, np.inf):
                    g[nxt] = ng
                    came_from[nxt] = cur
                    heapq.heappush(open_set, (ng + h(nxt), ng, nxt))
        return []

    def lane_at(self, route: list[RouteNode], s: float) -> int | None:
        """Lane the route occupies at arc length ``s`` (last node beyond the end)."""
        if not route:
            return None
        seg = int(s // self.seg_len)
        for n in route:
            if n.seg >= seg:
                return n.lane
        return route[-1].lane

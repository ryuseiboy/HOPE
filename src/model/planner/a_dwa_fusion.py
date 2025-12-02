import math
import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, box

from configs import VALID_SPEED, VALID_STEER, WHEEL_BASE, STEP_LENGTH


def _normalize_action(val: float, low: float, high: float) -> float:
    """Scale value in [low, high] to [-1, 1] for the env wrapper."""
    return 2 * (np.clip(val, low, high) - low) / (high - low) - 1


@dataclass(order=True)
class _AStarNode:
    f: float
    g: float
    h: float
    pos: Tuple[int, int]
    parent: Optional["__class__"] = None


class AStarPlanner:
    def __init__(self, resolution: float = 1.0, inflate_radius: float = 0.5):
        self.resolution = resolution
        self.inflate_radius = inflate_radius

    def _grid_from_map(self, parking_map) -> Tuple[np.ndarray, Tuple[float, float]]:
        xmin = math.floor(parking_map.xmin)
        ymin = math.floor(parking_map.ymin)
        xmax = math.ceil(parking_map.xmax)
        ymax = math.ceil(parking_map.ymax)

        width = int((xmax - xmin) / self.resolution) + 1
        height = int((ymax - ymin) / self.resolution) + 1
        grid = np.zeros((width, height), dtype=np.uint8)

        # Inflate obstacles slightly to be conservative
        inflated_obstacles: List[Polygon] = []
        for obs in parking_map.obstacles:
            inflated_obstacles.append(obs.shape.buffer(self.inflate_radius))

        for i in range(width):
            for j in range(height):
                cx = xmin + i * self.resolution
                cy = ymin + j * self.resolution
                cell = box(cx - self.resolution / 2, cy - self.resolution / 2,
                           cx + self.resolution / 2, cy + self.resolution / 2)
                for poly in inflated_obstacles:
                    if poly.intersects(cell):
                        grid[i, j] = 1
                        break
        return grid, (xmin, ymin)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], parking_map) -> Optional[List[Tuple[float, float]]]:
        grid, origin = self._grid_from_map(parking_map)
        start_idx = (int((start[0] - origin[0]) / self.resolution),
                     int((start[1] - origin[1]) / self.resolution))
        goal_idx = (int((goal[0] - origin[0]) / self.resolution),
                    int((goal[1] - origin[1]) / self.resolution))

        open_set = []
        g0 = 0.0
        h0 = self._heuristic(start_idx, goal_idx)
        heapq.heappush(open_set, _AStarNode(g0 + h0, g0, h0, start_idx, None))
        closed = set()

        parents = {}
        g_scores = {start_idx: 0.0}

        while open_set:
            current = heapq.heappop(open_set)
            if current.pos in closed:
                continue
            closed.add(current.pos)

            if current.pos == goal_idx:
                # reconstruct
                path = []
                node = current
                while node:
                    x = origin[0] + node.pos[0] * self.resolution
                    y = origin[1] + node.pos[1] * self.resolution
                    path.append((x, y))
                    node = node.parent
                path.reverse()
                return path

            x, y = current.pos
            neighbors = [
                (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)
            ]
            for nx, ny in neighbors:
                if nx < 0 or ny < 0 or nx >= grid.shape[0] or ny >= grid.shape[1]:
                    continue
                if grid[nx, ny] == 1:
                    continue
                tentative_g = current.g + math.hypot(nx - x, ny - y)
                # Dynamic weighting factor P
                denom = self._heuristic(start_idx, goal_idx) + 1e-6
                P = (abs(nx - goal_idx[0]) + abs(ny - goal_idx[1])) / denom
                h = math.exp(P) * self._heuristic((nx, ny), goal_idx)
                if (nx, ny) not in g_scores or tentative_g < g_scores[(nx, ny)]:
                    g_scores[(nx, ny)] = tentative_g
                    node = _AStarNode(tentative_g + h, tentative_g, h, (nx, ny), current)
                    heapq.heappush(open_set, node)
        return None

    def extract_key_nodes(self, path: List[Tuple[float, float]], stride: int = 2) -> List[Tuple[float, float]]:
        if not path:
            return []
        key_nodes = [path[0]]
        prev_dir = None
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            direction = (np.sign(dx), np.sign(dy))
            if prev_dir is None:
                prev_dir = direction
            if direction != prev_dir or i % stride == 0:
                key_nodes.append(path[i])
                prev_dir = direction
        if key_nodes[-1] != path[-1]:
            key_nodes.append(path[-1])
        return key_nodes


class DWAPlanner:
    def __init__(self, alpha: float = 15.0, beta: float = 37.0, gamma: float = 0.02):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _sample_window(self, v_curr: float, w_curr: float, v_min: float, v_max: float, w_max: float):
        # Simple sampling without acceleration limits
        v_samples = np.linspace(v_min, v_max, num=5)
        w_samples = np.linspace(-w_max, w_max, num=7)
        return v_samples, w_samples

    def _score(self, v: float, w: float, heading_error: float, dist_obs: float, dist_goal: float) -> float:
        vel_term = v
        head_term = math.cos(heading_error)
        dist_term = dist_obs  # larger is better
        return self.alpha * vel_term + self.beta * head_term + self.gamma * dist_term - 0.01 * dist_goal

    def plan(self, pose: Tuple[float, float, float], subgoal: Tuple[float, float], lidar_min: float) -> Tuple[float, float]:
        x, y, heading = pose
        goal_x, goal_y = subgoal
        dist_goal = math.hypot(goal_x - x, goal_y - y)
        heading_error = math.atan2(goal_y - y, goal_x - x) - heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        v_curr = 0.0
        w_curr = 0.0
        v_min, v_max = VALID_SPEED[0], VALID_SPEED[1]
        # angular velocity bound from steer limit
        w_max = abs(math.tan(VALID_STEER[1]) * VALID_SPEED[1] / WHEEL_BASE)

        v_samples, w_samples = self._sample_window(v_curr, w_curr, v_min, v_max, w_max)
        best_score = -float("inf")
        best = (0.0, 0.0)
        dt = STEP_LENGTH
        for v in v_samples:
            for w in w_samples:
                # simulate one step
                x1 = x + v * math.cos(heading) * dt
                y1 = y + v * math.sin(heading) * dt
                heading1 = heading + w * dt
                heading_err1 = math.atan2(goal_y - y1, goal_x - x1) - heading1
                heading_err1 = math.atan2(math.sin(heading_err1), math.cos(heading_err1))
                score = self._score(v, w, heading_err1, lidar_min, dist_goal)
                if score > best_score:
                    best_score = score
                    best = (v, w)
        return best


class ADWAFusionPlanner:
    """
    A* (global) + DWA (local) fusion planner.
    Provides one-step action (steer, speed) scaled to env raw action range [-1, 1].
    """
    def __init__(self, resolution: float = 1.0, inflate_radius: float = 0.5):
        self.astar = AStarPlanner(resolution, inflate_radius)
        self.dwa = DWAPlanner()
        self.key_nodes: List[Tuple[float, float]] = []
        self.current_idx = 0

    def reset(self):
        self.key_nodes = []
        self.current_idx = 0

    def _ensure_path(self, pose: Tuple[float, float, float], goal: Tuple[float, float], parking_map):
        need_plan = False
        if not self.key_nodes:
            need_plan = True
        elif self.current_idx >= len(self.key_nodes):
            need_plan = True
        if need_plan:
            path = self.astar.plan((pose[0], pose[1]), goal, parking_map)
            if path is None:
                self.reset()
                return
            self.key_nodes = self.astar.extract_key_nodes(path)
            self.current_idx = 0

    def _advance_if_reached(self, pose: Tuple[float, float, float], threshold: float = 0.5):
        if not self.key_nodes or self.current_idx >= len(self.key_nodes):
            return
        x, y, _ = pose
        gx, gy = self.key_nodes[self.current_idx]
        if math.hypot(gx - x, gy - y) < threshold:
            self.current_idx += 1

    def plan_action(self, pose: Tuple[float, float, float], goal: Tuple[float, float], parking_map, lidar_min: float) -> np.ndarray:
        # refresh global/subgoals if needed
        self._ensure_path(pose, goal, parking_map)
        self._advance_if_reached(pose)

        # fallback: go straight to goal if no path
        if not self.key_nodes:
            target = goal
        else:
            idx = min(self.current_idx, len(self.key_nodes) - 1)
            target = self.key_nodes[idx]

        v, w = self.dwa.plan(pose, target, lidar_min)
        # convert (v, w) to (steer, speed)
        speed = np.clip(v, VALID_SPEED[0], VALID_SPEED[1])
        steer = 0.0
        if abs(speed) > 1e-3:
            steer = math.atan2(w * WHEEL_BASE, speed)
        steer = np.clip(steer, VALID_STEER[0], VALID_STEER[1])

        raw_steer = _normalize_action(steer, VALID_STEER[0], VALID_STEER[1])
        raw_speed = _normalize_action(speed, VALID_SPEED[0], VALID_SPEED[1])
        return np.array([raw_steer, raw_speed], dtype=np.float32)

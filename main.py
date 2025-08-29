
import os
import sys
import threading
import time
import math
from pathlib import Path
import argparse
import pygame

# --- Configuration defaults ---
DEFAULT_MAX_NODES = 800
SCAN_SLEEP = 0.01  # small sleep between scandir entries to make scanning incremental
FORCE_ITER_PER_FRAME = 1  # how many physics iterations per render frame
# -------------------------------

class FSGraph:
    def __init__(self, root_path, follow_symlinks=False, max_nodes=DEFAULT_MAX_NODES):
        self.root = Path(root_path)
        self.follow_symlinks = follow_symlinks
        self.max_nodes = max_nodes
        # nodes: path_str -> node dict
        self.nodes = {}
        # edges: list of (parent_path_str, child_path_str)
        self.edges = []
        self.lock = threading.RLock()
        self._stop = False
        # physics state
        self.positions = {}   # path -> [x,y]
        self.velocities = {}  # path -> [vx,vy]
        self.fixed = set()    # paths that should not move (not used much)
        # scanning control
        self.paused = False

        # initialize root node
        self._add_node(str(self.root), parent=None, node_type='dir')

    def _add_node(self, path_str, parent=None, node_type='file'):
        if path_str in self.nodes:
            return False
        if len(self.nodes) >= self.max_nodes:
            return False
        node = {
            'path': path_str,
            'name': os.path.basename(path_str) or path_str,
            'type': node_type,
            'children': [],
            'note': None,
        }
        self.nodes[path_str] = node
        if parent:
            node['parent'] = parent
            self.edges.append((parent, path_str))
            self.nodes[parent]['children'].append(path_str)
        else:
            node['parent'] = None
        # init physics pos near parent or at origin
        if parent and parent in self.positions:
            px, py = self.positions[parent]
            # small random offset
            angle = (len(self.nodes) % 360) * 0.1
            r = 20 + (len(self.nodes) % 10) * 3
            self.positions[path_str] = [px + r * math.cos(angle), py + r * math.sin(angle)]
        else:
            # center root at 0,0
            self.positions[path_str] = [0.0, 0.0]
        self.velocities[path_str] = [0.0, 0.0]
        return True

    def scan_incremental(self):
        """
        Walk the filesystem incrementally and add nodes to the graph.
        Uses a stack for DFS to give a more 'growing branch' feel.
        """
        # We'll use lstat/stat to avoid infinite loops when not following symlinks.
        seen_inodes = set()
        stack = [Path(self.root)]
        while stack and not self._stop:
            if self.paused:
                time.sleep(0.1)
                continue
            p = stack.pop()
            p_str = str(p)
            try:
                st = p.stat() if self.follow_symlinks else p.lstat()
            except Exception:
                # inaccessible
                with self.lock:
                    if p_str not in self.nodes:
                        self._add_node(p_str, parent=None if p==self.root else str(p.parent), node_type='dir' if p.is_dir() else 'file')
                        self.nodes[p_str]['note'] = 'inaccessible'
                continue
            inode_id = (st.st_dev, st.st_ino) if st else None
            if inode_id and inode_id in seen_inodes:
                # already visited (hardlink/symlink)
                continue
            if inode_id:
                seen_inodes.add(inode_id)

            # Add node if not present
            parent = None if p == self.root else str(p.parent)
            with self.lock:
                self._add_node(p_str, parent=parent, node_type='dir' if p.is_dir() else 'file')

            # If directory, push children
            if p.is_dir() or (p.is_symlink() and self.follow_symlinks and p.resolve().is_dir()):
                try:
                    with os.scandir(p) as it:
                        entries = []
                        for entry in it:
                            entries.append(entry)
                        # sort entries to make scanning deterministic: directories first
                        entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=self.follow_symlinks), e.name.lower()))
                        # push in reverse so first entries get scanned first
                        for entry in reversed(entries):
                            if len(self.nodes) >= self.max_nodes:
                                break
                            child_p = Path(entry.path)
                            stack.append(child_p)
                            # add child node immediately for responsiveness
                            child_str = str(child_p)
                            with self.lock:
                                self._add_node(child_str, parent=p_str, node_type='dir' if entry.is_dir(follow_symlinks=self.follow_symlinks) else 'file')
                            time.sleep(SCAN_SLEEP)
                except PermissionError:
                    with self.lock:
                        self.nodes[p_str]['note'] = 'permission denied'
                except FileNotFoundError:
                    with self.lock:
                        self.nodes[p_str]['note'] = 'not found'
            # small sleep to yield so main thread can render updates
            time.sleep(SCAN_SLEEP)

    def stop(self):
        self._stop = True

    # --- Simple force-directed layout (spring + repulsion) ---
    def physics_step(self, area=10000, k=None, damping=0.85):
        """
        Update positions and velocities for all nodes.
        area parameter influences ideal edge length scaling.
        """
        with self.lock:
            n = max(1, len(self.positions))
            if k is None:
                k = math.sqrt(area / n)  # ideal distance between nodes
            # initialize forces
            forces = {p: [0.0, 0.0] for p in self.positions.keys()}

            # repulsive forces (O(n^2) naive) - OK for modest node counts
            items = list(self.positions.items())
            for i in range(len(items)):
                pi, (xi, yi) = items[i]
                for j in range(i+1, len(items)):
                    pj, (xj, yj) = items[j]
                    dx = xi - xj
                    dy = yi - yj
                    dist2 = dx*dx + dy*dy + 1e-6
                    dist = math.sqrt(dist2)
                    # repulsive force magnitude (Coulomb-like)
                    F = (k*k) / dist
                    fx = F * dx / dist
                    fy = F * dy / dist
                    forces[pi][0] += fx
                    forces[pi][1] += fy
                    forces[pj][0] -= fx
                    forces[pj][1] -= fy

            # attractive forces along edges (Hooke's law)
            for (a, b) in list(self.edges):
                if a not in self.positions or b not in self.positions:
                    continue
                xa, ya = self.positions[a]
                xb, yb = self.positions[b]
                dx = xa - xb
                dy = ya - yb
                dist = math.sqrt(dx*dx + dy*dy) + 1e-6
                # spring force proportional to distance - ideal length = k
                F = (dist*dist) / k
                fx = -F * dx / dist
                fy = -F * dy / dist
                forces[a][0] += fx
                forces[a][1] += fy
                forces[b][0] -= fx
                forces[b][1] -= fy

            # integrate velocities and positions
            for p, (x, y) in list(self.positions.items()):
                if p in self.fixed:
                    continue
                fx, fy = forces[p]
                vx, vy = self.velocities.get(p, [0.0, 0.0])
                # simple Euler integration
                vx = (vx + fx * 0.001) * damping
                vy = (vy + fy * 0.001) * damping
                self.velocities[p] = [vx, vy]
                self.positions[p][0] = x + vx
                self.positions[p][1] = y + vy

# --- Pygame visualization ---
def run_pygame(root, follow_symlinks=False, max_nodes=DEFAULT_MAX_NODES, width=1200, height=800):
    pygame.init()
    pygame.display.set_caption("Filesystem Real-time Graph Visualizer")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 16)
    big_font = pygame.font.SysFont(None, 22)

    graph = FSGraph(root, follow_symlinks=follow_symlinks, max_nodes=max_nodes)

    # Start scanning thread
    scanner_thread = threading.Thread(target=graph.scan_incremental, daemon=True)
    scanner_thread.start()

    # camera / interaction state
    cam_x, cam_y = width // 2, height // 2
    zoom = 1.0
    dragging = False
    last_mouse = (0,0)
    selected_node = None

    paused = False

    def world_to_screen(pos):
        x, y = pos
        sx = cam_x + x * zoom
        sy = cam_y + y * zoom
        return int(sx), int(sy)

    def screen_to_world(sx, sy):
        return ( (sx - cam_x) / zoom, (sy - cam_y) / zoom )

    running = True
    while running:
        dt = clock.tick(30) / 1000.0  # seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                graph.stop()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    dragging = True
                    last_mouse = event.pos
                    # check if clicked on a node (closest within radius)
                    mx, my = event.pos
                    wx, wy = screen_to_world(mx, my)
                    # find nearest node within threshold
                    with graph.lock:
                        nearest = None
                        ndist = 1e9
                        for p, pos in graph.positions.items():
                            dx = pos[0] - wx
                            dy = pos[1] - wy
                            dist = math.hypot(dx, dy)
                            if dist < ndist:
                                ndist = dist
                                nearest = p
                        if nearest and ndist * zoom < 12:
                            selected_node = nearest
                            # center camera on node
                            px, py = graph.positions[nearest]
                            cam_x = width//2 - px*zoom
                            cam_y = height//2 - py*zoom
                elif event.button == 3:  # right click
                    # toggle scanner pause/resume
                    graph.paused = not graph.paused
                elif event.button == 4:  # wheel up
                    # zoom in (centered at mouse)
                    mx, my = event.pos
                    wx_before = (mx - cam_x) / zoom
                    zoom *= 1.1
                    wx_after = (mx - cam_x) / zoom
                    cam_x += (wx_after - wx_before) * zoom
                elif event.button == 5:  # wheel down
                    mx, my = event.pos
                    wx_before = (mx - cam_x) / zoom
                    zoom /= 1.1
                    wx_after = (mx - cam_x) / zoom
                    cam_x += (wx_after - wx_before) * zoom
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    lx, ly = last_mouse
                    dx = mx - lx
                    dy = my - ly
                    cam_x += dx
                    cam_y += dy
                    last_mouse = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    graph.paused = not graph.paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    graph.max_nodes += 100
                elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                    graph.max_nodes = max(100, graph.max_nodes - 100)

        # physics steps
        for _ in range(FORCE_ITER_PER_FRAME):
            graph.physics_step(area=20000)

        # draw
        screen.fill((24, 24, 30))
        # draw edges
        with graph.lock:
            for a, b in graph.edges:
                if a in graph.positions and b in graph.positions:
                    ax, ay = world_to_screen(graph.positions[a])
                    bx, by = world_to_screen(graph.positions[b])
                    # lighter line for deeper edges
                    pygame.draw.aaline(screen, (80,80,100), (ax, ay), (bx, by))

            # draw nodes (directories larger)
            for p, pos in graph.positions.items():
                x, y = world_to_screen(pos)
                node = graph.nodes.get(p, {})
                ntype = node.get('type', 'file')
                if ntype == 'dir':
                    r = int(8 * zoom) + 2
                    color = (200, 160, 60)
                else:
                    r = max(2, int(3 * zoom))
                    color = (150, 180, 220)
                pygame.draw.circle(screen, color, (x, y), max(1, r))
                # highlight selected
                if p == selected_node:
                    pygame.draw.circle(screen, (255,80,80), (x, y), max(1, r+3), 2)

            # labels: show shallow nodes + selected node
            height = screen.get_height()
            label_count = 0
            for p, pos in sorted(graph.positions.items(), key=lambda kv: (len(kv[0].split(os.sep)), kv[0])):
                if label_count > 200:
                    break
                depth = len(p.split(os.sep)) - 1
                # label shallow or selected or near center
                if depth <= 2 or p == selected_node:
                    x, y = world_to_screen(pos)
                    name = graph.nodes.get(p, {}).get('name', os.path.basename(p))
                    txt = font.render(name, True, (220,220,220))
                    screen.blit(txt, (x + 6, y + 2))
                    label_count += 1

        # HUD
        hud_lines = [
            f"Root: {root}  nodes: {len(graph.nodes)}  edges: {len(graph.edges)}  max_nodes: {graph.max_nodes}  paused: {graph.paused}",
            "Controls: drag to pan, wheel to zoom, left-click node to center, right-click or Space to pause scanner"
        ]
        y = 6
        for line in hud_lines:
            surf = big_font.render(line, True, (200,200,200))
            screen.blit(surf, (6, y))
            y += surf.get_height() + 2

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filesystem real-time graph visualizer (pygame)")
    parser.add_argument("root", nargs="?", help="Root folder to scan (default: current directory)", default=".")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow directory symlinks (careful: may create cycles)")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES, help="Maximum number of nodes to scan/display")
    args = parser.parse_args()
    root = args.root
    follow = args.follow_symlinks
    max_nodes = args.max_nodes
    run_pygame(root, follow_symlinks=follow, max_nodes=max_nodes)

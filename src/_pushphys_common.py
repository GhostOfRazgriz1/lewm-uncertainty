"""Real-physics 2D pusher (pymunk) -- the non-toy substrate. Genuine contact, friction, rotation
(Chipmunk2D), NOT a hand-coded shove rule. State-based, headless. The expert/oracle is MPC over the
TRUE simulator (privileged), so we get a competent controller without scripting a push heuristic.

state s = [pusher_x, pusher_y, block_x, block_y, cos(theta), sin(theta)] / arena   (6-d, theta = block angle)
action a = desired pusher velocity in [-1,1]^2 (scaled). Quasi-static (heavy damping) so positions ~Markov.
events: 1 contact, 2 moved, 3 delivered (block within tol of goal). T-block optional (block='T').
"""
import numpy as np
import pymunk

W = 500.0
DT = 1.0 / 50.0
SUBSTEPS = 4
PVEL = 230.0                       # pusher max speed (px/s)
DAMP = 0.25                        # space damping (heavy -> quasi-static pushing)
PUSH_R = 14.0


def _moment_box(m, w, h):
    return pymunk.moment_for_box(m, (w, h))


class PushPhysEnv:
    def __init__(self, goal=(300.0, 250.0), block="box", pos_tol=50.0, behind_start=True):
        self.goal = np.array(goal, "float32"); self.block_kind = block; self.pos_tol = pos_tol
        self.behind_start = behind_start
        self._build()

    def _build(self):
        sp = pymunk.Space(); sp.gravity = (0, 0); sp.damping = DAMP
        for a, b in [((25, 25), (W - 25, 25)), ((W - 25, 25), (W - 25, W - 25)),
                     ((W - 25, W - 25), (25, W - 25)), ((25, W - 25), (25, 25))]:
            seg = pymunk.Segment(sp.static_body, a, b, 4); seg.friction = 0.7; seg.elasticity = 0.1; sp.add(seg)
        pm = 2.0
        self.pusher = pymunk.Body(pm, pymunk.moment_for_circle(pm, 0, PUSH_R))
        ps = pymunk.Circle(self.pusher, PUSH_R); ps.friction = 0.9; ps.elasticity = 0.05
        sp.add(self.pusher, ps)
        bm = 1.0
        if self.block_kind == "box":
            self.block = pymunk.Body(bm, _moment_box(bm, 74, 74))
            self.bshapes = [pymunk.Poly.create_box(self.block, (74, 74))]
        else:                                                          # T-block: horizontal bar + downward stem
            self.block = pymunk.Body(bm, _moment_box(bm, 100, 100))
            self.bshapes = [pymunk.Poly.create_box(self.block, (100, 26)),
                            pymunk.Poly(self.block, [(-13, -13), (13, -13), (13, -70), (-13, -70)])]
        for s in self.bshapes:
            s.friction = 0.9; s.elasticity = 0.03
        sp.add(self.block, *self.bshapes)
        self.space = sp

    def reset(self, g):
        while True:
            bp = g.uniform(110, W - 110, 2)
            if np.linalg.norm(bp - self.goal) > self.pos_tol + 70:
                break
        if self.behind_start:                                          # pusher just behind the block w.r.t. the goal
            bg = (self.goal - bp) / (np.linalg.norm(self.goal - bp) + 1e-9)
            pp = bp - bg * (PUSH_R + 55) + g.uniform(-18, 18, 2)
            pp = np.clip(pp, 40, W - 40)
        else:
            while True:
                pp = g.uniform(60, W - 60, 2)
                if np.linalg.norm(pp - bp) > 70:
                    break
        self.pusher.position = tuple(pp); self.pusher.velocity = (0, 0)
        self.block.position = tuple(bp); self.block.velocity = (0, 0)
        self.block.angle = float(g.uniform(-0.4, 0.4)); self.block.angular_velocity = 0.0
        self.space.step(1e-4)
        return self._state()

    def _state(self):
        p = self.pusher.position; b = self.block.position; th = self.block.angle
        return np.array([p.x / W, p.y / W, b.x / W, b.y / W, np.cos(th), np.sin(th)], "float32")

    def _contact(self):
        d = min(s.point_query(self.pusher.position).distance for s in self.bshapes)
        return d < PUSH_R + 3.0

    def step(self, a):
        self.pusher.velocity = (float(np.clip(a[0], -1, 1)) * PVEL, float(np.clip(a[1], -1, 1)) * PVEL)
        b0 = np.array(self.block.position)
        for _ in range(SUBSTEPS):
            self.space.step(DT / SUBSTEPS)
        s = self._state(); bpos = np.array(self.block.position)
        moved = np.linalg.norm(bpos - b0) > 2.5
        delivered = np.linalg.norm(bpos - self.goal) < self.pos_tol
        ev = 3 if delivered else (2 if moved else (1 if self._contact() else 0))
        return s, ev

    # ---- privileged save/restore for true-sim MPC ----
    def get_full(self):
        p, b = self.pusher, self.block
        return (tuple(p.position), tuple(p.velocity), tuple(b.position), tuple(b.velocity), b.angle, b.angular_velocity)

    def set_full(self, fs):
        pp, pv, bp, bv, ba, bav = fs
        self.pusher.position = pp; self.pusher.velocity = pv
        self.block.position = bp; self.block.velocity = bv; self.block.angle = ba; self.block.angular_velocity = bav

    def block_xy(self):
        return np.array(self.block.position, "float32")


def oracle_action(env, rng, N=40, H=12, iters=2):
    """Competent oracle = the scripted push controller SEEDED into a short true-simulator CEM. The script gives
    the navigate-behind-then-push structure (so the goal cost is no longer flat); the true physics refines it
    (rotation, walls). Returns the first action of the best plan."""
    fs = env.get_full()
    seed = np.zeros((H, 2), "float32")                                # scripted rollout under true physics = seed
    for h in range(H):
        seed[h] = scripted_push(env); env.step(seed[h])
    env.set_full(fs)
    mean, std, best = seed.copy(), np.ones((H, 2)) * 0.4, seed[0].copy()
    for _ in range(iters):
        seqs = np.clip(mean + std * rng.standard_normal((N, H, 2)), -1, 1).astype("float32"); seqs[0] = seed
        costs = np.empty(N)
        for i in range(N):
            env.set_full(fs)
            for h in range(H):
                env.step(seqs[i, h])
            costs[i] = np.linalg.norm(env.block_xy() - env.goal)
        order = costs.argsort(); elite = seqs[order[:max(1, N // 4)]]
        mean, std, best = elite.mean(0), elite.std(0) + 0.05, seqs[order[0], 0]
    env.set_full(fs)
    return best


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)


def scripted_push(env):
    """Competent feedback pusher for the real physics: get behind the block w.r.t. the goal (swinging AROUND
    it if on the goal side), then push toward the goal. Serves as the oracle upper bound and DAgger expert."""
    p = np.array(env.pusher.position, "float32"); b = env.block_xy(); g = env.goal
    bg = _u(g - b); behind = b - bg * (PUSH_R + 16.0)
    if np.linalg.norm(p - behind) < 26.0:
        return _u(g - p)                                           # in pushing position -> push toward goal
    if np.dot(p - b, bg) > -5.0:                                   # pusher on the goal side -> swing around
        perp = np.array([-bg[1], bg[0]], "float32")
        side = perp if np.dot(p - b, perp) >= 0 else -perp
        return _u(b + side * (PUSH_R + 60.0) - bg * 12.0 - p)
    return _u(behind - p)                                          # clear path -> go behind


def rollout_success(env, policy, g, episodes, T_plan):
    sc = 0
    for _ in range(episodes):
        env.reset(g)
        for t in range(T_plan):
            s = env._state()
            _, _ = env.step(policy(s))
            if np.linalg.norm(env.block_xy() - env.goal) < env.pos_tol:
                sc += 1; break
    return sc / episodes

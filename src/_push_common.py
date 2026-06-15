"""Minimal PUSHER env for the scale-up de-risk: reachability = structured pushing, not navigation.

State [agent_x, agent_y, block_x, block_y] (4-d), action (dx,dy), deterministic, from state. Walking into
the block shoves it ahead of you -> to deliver the block to the goal you must approach from the side OPPOSITE
the goal and push (the toy's 'go to the object' no longer works). Events are EMERGENT from the push dynamics
(CONTACT / MOVED / DELIVERED), with ground-truth labels for eval only. See docs/scaleup-pusher-spec.md.
"""
import numpy as np

ADIM = 2
PSDIM = 4
PNEV = 4
P_ENAMES = ["none", "contact", "moved", "delivered"]
PSTEP = 0.05


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)


class PushEnv:
    def __init__(self, goal=(0.85, 0.5), contact_r=0.10, zone_r=0.12, move_thresh=0.008):
        self.goal = np.array(goal, "float32"); self.contact_r = contact_r; self.zone_r = zone_r; self.move_thresh = move_thresh

    def reset(self, g):
        while True:
            agent = g.uniform(0.1, 0.9, 2); block = g.uniform(0.15, 0.85, 2)
            if np.linalg.norm(block - self.goal) > self.zone_r + 0.08 and np.linalg.norm(agent - block) > self.contact_r + 0.02:
                return np.concatenate([agent, block]).astype("float32")

    def step(self, s, a):
        agent = np.clip(s[:2] + a, 0, 1).astype("float32")
        block = s[2:4].copy()
        d = block - agent; dist = np.linalg.norm(d)
        if dist < self.contact_r:
            block = np.clip(agent + self.contact_r * _u(d), 0, 1).astype("float32")   # shoved ahead of agent
        disp = np.linalg.norm(block - s[2:4])
        deliv = np.linalg.norm(block - self.goal) < self.zone_r
        deliv_before = np.linalg.norm(s[2:4] - self.goal) < self.zone_r
        touching = np.linalg.norm(agent - block) < self.contact_r + 1e-6
        if deliv and not deliv_before:
            ev = 3
        elif disp > self.move_thresh:
            ev = 2
        elif touching:
            ev = 1
        else:
            ev = 0
        return np.concatenate([agent, block]).astype("float32"), ev

    def expert(self, s, e):                                            # scripted skill toward target event
        agent, block = s[:2], s[2:4]
        if e == 1:                                                     # CONTACT: approach the block (navigation)
            return np.clip(_u(block - agent) * PSTEP, -PSTEP, PSTEP).astype("float32")
        behind = block + self.contact_r * _u(block - self.goal)        # MOVED/DELIVERED: get behind, then push
        target = behind if np.linalg.norm(agent - behind) > self.contact_r * 0.6 else self.goal
        return np.clip(_u(target - agent) * PSTEP, -PSTEP, PSTEP).astype("float32")

    def policy(self, s, g):                                            # mixed data-collection policy
        r = g.random()
        if r < 0.35:
            return (g.uniform(-1, 1, 2) * PSTEP).astype("float32")
        return self.expert(s, 1 if r < 0.55 else 3)

    def collect(self, g, n_ep, T):
        Z, A, Zn, EV, rolls = [], [], [], [], []
        for _ in range(n_ep):
            s = self.reset(g); ep = [s]
            for _ in range(T):
                a = self.policy(s, g); s2, ev = self.step(s, a)
                Z.append(s); A.append(a); Zn.append(s2); EV.append(ev); ep.append(s2); s = s2
            rolls.append(np.array(ep))
        return (np.array(Z, "float32"), np.array(A, "float32"), np.array(Zn, "float32"), np.array(EV)), rolls

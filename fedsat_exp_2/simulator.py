
from dataclasses import dataclass
from typing import List, Callable
import heapq

MINUTE = 60.0

@dataclass
class Satellite:
    sid: int
    period_min: float
    phase_min: float
    contact_duration_min: float
    local_data_size: int
    name: str = ""
    vis_stride: int = 1  # <-- NEW: only every k-th orbit is visible    
    shell: int = 0  # <--   add  this
    
    def next_pass_start(self, k: int) -> float:
        return (self.phase_min + k * self.period_min) * MINUTE
    def next_pass_end(self, k: int) -> float:
        return self.next_pass_start(k) + self.contact_duration_min * MINUTE

class EventLoop:
    def __init__(self, sats: List[Satellite], horizon_min: float):
        self.sats = sats
        self.horizon_s = horizon_min * MINUTE
        self.pq = []
        for s in sats:
            k = 0
            while True:
                t0 = s.next_pass_start(k); t1 = s.next_pass_end(k)
                if t0 > self.horizon_s: break
                # NEW: only schedule if this pass is visible
                if (k % s.vis_stride) == 0:
                    heapq.heappush(self.pq, (t0, "contact_start", s.sid, k))
                    if t1 <= self.horizon_s:
                        heapq.heappush(self.pq, (t1, "contact_end", s.sid, k))
                k += 1


    def run(self, on_event: Callable[[float, str, int, int], None]):
        while self.pq:
            t, etype, sid, k = heapq.heappop(self.pq)
            on_event(t, etype, sid, k)

def preset_bremen_two_shells(n_per_shell=(5,5), periods=(97.0,127.0),
                             phases=(0.0,0.0), contact_duration=8.0,
                             data_per_sat=12000,
                             vis_stride_shell=(1,1)):
    sats = []
    sid = 0
    for shell_idx, n in enumerate(n_per_shell):
        stride = vis_stride_shell[shell_idx]
        for i in range(n):
            sats.append(Satellite(
                sid=sid,
                period_min=periods[shell_idx],
                phase_min=phases[shell_idx] + i * (periods[shell_idx] / n),
                contact_duration_min=contact_duration,
                local_data_size=data_per_sat,
                name=f"shell{shell_idx}_sat{i}",
                vis_stride=stride,
                shell=shell_idx,  # ✅ FIX: assign shell group properly
            ))
            sid += 1
    return sats



def preset_pole_single_shell(n=6, period=100.0, phase0=0.0, contact_duration=10.0, data_per_sat=10000):
    sats = []
    for i in range(n):
        sats.append(Satellite(
            sid=i, period_min=period, phase_min=phase0 + i*(period/n),
            contact_duration_min=contact_duration, local_data_size=data_per_sat,
            name=f"polar_sat{i}"
        ))
    return sats

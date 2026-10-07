import sys, time
from avstack.runner import run
from avstack.sim.scenarios import SCENARIOS
names = sys.argv[1:] or list(SCENARIOS)
for n in names:
    t0 = time.time()
    r = run(SCENARIOS[n]())
    print(r.summary(), f"\n  wall={time.time()-t0:.1f}s")
    from collections import Counter
    print("  behaviors:", Counter(r.history["behavior"]).most_common())

"""Check that offloading the frame cache leaves outputs unchanged: rerun KITTI-MOTS 0014 with the
offload and compare with the benchmark file (produced without it) byte for byte."""
import filecmp, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import run_sam_mots as r

model = sys.argv[1]
tracker = "SAM3" if model == "sam3" else "SAM3.1"
bench = r.BENCHMARKS["kitti_mots"]
frames = sorted(p for p in bench["frames"]("0014").iterdir() if p.suffix == ".png")
from PIL import Image
w, h = Image.open(frames[0]).size
pred, _ = r.build(model)
out = Path(f"/tmp/verify_offload/{tracker}/0014.txt"); out.parent.mkdir(parents=True, exist_ok=True)
torch.cuda.reset_peak_memory_stats(); t = time.time()
r.run_sequence(pred, bench, bench["frames"]("0014"), frames, out, h, w, tracker, "verify", "0014")
ref = Path(f"trackers/kitti_mots/{tracker}/data/0014.txt")
same = filecmp.cmp(out, ref, shallow=False)
print(f"VERIFY {tracker} 0014 identical={same} peak={torch.cuda.max_memory_reserved()/2**30:.1f}GB {time.time()-t:.0f}s", flush=True)

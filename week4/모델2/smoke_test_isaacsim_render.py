"""Decisive test: does Isaac Sim's RTX renderer produce a real frame on this
RTX 5070 Ti (Blackwell)?  This mirrors the Week 4 egocentric-capture path
(omni.replicator rgb annotator on a camera), so a PASS here means closed-loop
frame capture will work; a black/crash means the GPU/renderer combo is the
blocker.

Run (EULA auto-accepted):
    set OMNI_KIT_ACCEPT_EULA=YES
    python smoke_test_isaacsim_render.py
"""

import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

# headless, but RTX renderer enabled so we actually render.
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# ground plane + dome light (guaranteed illumination) + a bright cube.
UsdGeom.Xform.Define(stage, "/World")
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -1.05))
ground.AddScaleOp().Set(Gf.Vec3d(20, 20, 0.05))
ground.GetDisplayColorAttr().Set([(0.4, 0.4, 0.4)])

dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1500.0)
distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
distant.CreateIntensityAttr(3000.0)

cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
cube.GetDisplayColorAttr().Set([(0.9, 0.15, 0.15)])

# replicator camera with explicit look_at -> guaranteed framing; 512 >= DLSS min.
cam = rep.create.camera(position=(0, -6, 2), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, (512, 512))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

# warm up the renderer (more steps so RTX accumulation/denoise settles)
for _ in range(80):
    rep.orchestrator.step(rt_subframes=8)
    app.update()

data = np.asarray(rgb.get_data())
os.makedirs("data", exist_ok=True)
verdict_path = os.path.join("data", "isaac_render_verdict.txt")
png_path = os.path.join("data", "isaac_render_test.png")

lines = [f"frame shape={data.shape} dtype={data.dtype}"]
ok = False
if data.ndim >= 3 and data.shape[-1] >= 3:
    arr = data[..., :3].astype(np.float32)
    lines.append(f"mean={arr.mean():.2f} min={arr.min():.0f} max={arr.max():.0f} "
                 f"std={arr.std():.2f} nonzero_frac={(arr > 5).mean():.3f}")
    ok = arr.max() > 10 and arr.std() > 2.0
    try:
        from PIL import Image
        Image.fromarray(data[..., :3].astype(np.uint8)).save(png_path)
        lines.append(f"saved PNG -> {png_path}")
    except Exception as exc:
        lines.append(f"PNG save failed: {exc}")
lines.append("VERDICT: " + ("PASS (non-black frame rendered on Blackwell)" if ok else "FAIL (black/flat/no frame)"))

with open(verdict_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
for ln in lines:
    print("[RESULT]", ln)

app.close()
print("[DONE] Isaac Sim render smoke test finished.")

import os
from src.data.synth_generator import generate

out = "data/test_synth"
if os.path.exists(out):
    import shutil; shutil.rmtree(out)

generate(n_patients=10, days=60, out_dir=out)

assert os.path.exists(os.path.join(out,"metadata.csv"))
assert os.path.exists(os.path.join(out,"labs.csv"))


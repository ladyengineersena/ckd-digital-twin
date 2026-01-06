from src.data.synth_generator import generate
from src.preprocess.features import load_data, build_sequence_data
import os

out = "data/test_sim2"
if os.path.exists(out):
    import shutil; shutil.rmtree(out)

generate(n_patients=5, days=120, out_dir=out)
meta, labs, meds, events = load_data(out)
seqs, tabs, pids = build_sequence_data(meta, labs, seq_len=30)

assert seqs.shape[0] == 5


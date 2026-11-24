import pandas as pd, os, sys
D="/fs/gamma-projects/audio/audio_datasets/audioset_strong"
names=dict(l.rstrip("\n").split("\t") for l in open(f"{D}/mid_to_display_name.tsv"))
have={f[:-4] for f in os.listdir(f"{D}/audio_clips_train")}
df=pd.read_csv(f"{D}/audioset_train_strong.tsv",sep="\t")
df=df[df.segment_id.isin(have)]
df["dur"]=df.end_time_seconds-df.start_time_seconds
# merged coverage per (clip,label) approx = sum durations clipped to 10
cov=df.groupby(["segment_id","label"]).dur.sum().clip(upper=10).reset_index()
nlab=df.groupby("segment_id").label.nunique()
cov["nlab"]=cov.segment_id.map(nlab)
good=cov[(cov.dur>=8.5)&(cov.nlab<=3)]
s=good.groupby("label").size().sort_values(ascending=False)
for m,c in s.head(60).items(): print(c, m, names.get(m))

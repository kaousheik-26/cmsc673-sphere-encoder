R=/fs/gamma-projects/audio/kajayaku/sphere_encoder_2/se2_audio_text
export HF_HOME=$R/hf TORCH_HOME=$R/torch_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 PYTHONUNBUFFERED=1
export PATH=$R/.venv/bin:$PATH
cd $R

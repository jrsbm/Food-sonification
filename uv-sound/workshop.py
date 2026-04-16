import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    app = mo.App(width="medium")
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import soundfile as sf
    from scipy.signal import savgol_filter

    return librosa, mo, np, pl, plt, savgol_filter


@app.cell
def _(pl):
    df = pl.read_csv('Instant_Coffee_Test_Samples.csv')
    sample_options = {f"ID: {df[i, 'Sample_ID']} (Row {i})": i 
                      for i in range(len(df))} 
    df
    return df, sample_options


@app.cell
def _(mo, sample_options):
    # --- UI Elements ---
    sample_dropdown = mo.ui.dropdown(
        options=sample_options, 
        value='ID: 1 (Row 0)', 
        label="Coffee Sample"
    )

    stretch_slider = mo.ui.slider(
        start=0.2, stop=4.0, step=0.01, value=1.0, label="Pitch Shift"
    )

    smooth_slider = mo.ui.slider(
        start=5, stop=51, step=2, value=15, label="Smoothness"
    )

    volume_slider = mo.ui.slider(
        start=0, stop=1.0, step=0.1, value=0.8, label="Master Volume"
    )

    # ADSR Sliders
    attack = mo.ui.slider(0.01, 0.5, step=0.01, value=0.1, label="A")
    decay = mo.ui.slider(0.01, 0.5, step=0.01, value=0.2, label="D")
    sustain = mo.ui.slider(0.1, 1.0, step=0.1, value=0.7, label="S")
    release = mo.ui.slider(0.1, 1.0, step=0.1, value=0.5, label="R")

    adsr_group = mo.hstack([attack, decay, sustain, release], justify="start")

    # Layout the UI
    ui = mo.vstack([
        mo.md("# ☕ Coffee Sonification Lab"),
        sample_dropdown,
        mo.hstack([stretch_slider, volume_slider, smooth_slider]),
        mo.md("**ADSR Envelope Settings**"),
        adsr_group
    ]).style({"background-color": "#f9f9f9", "padding": "20px", "border-radius": "10px"})

    ui
    return (
        attack,
        decay,
        release,
        sample_dropdown,
        smooth_slider,
        stretch_slider,
        sustain,
        volume_slider,
    )


@app.cell
def _(np):
    ## Envelope
    def create_adsr_envelope(duration, sr, a, d, s, r):
        total_samples = int(duration * sr)
        a_samples = int(a * sr)
        d_samples = int(d * sr)
        r_samples = int(r * sr)
        s_samples = total_samples - (a_samples + d_samples + r_samples)

        if s_samples < 0:
            return np.zeros(total_samples) # Safety check

        envelope = np.concatenate([
            np.linspace(0, 1, a_samples),
            np.linspace(1, s, d_samples),
            np.full(s_samples, s),
            np.linspace(s, 0, r_samples)
        ])
        return envelope

    return (create_adsr_envelope,)


@app.cell
def _(
    attack,
    create_adsr_envelope,
    decay,
    df,
    librosa,
    mo,
    np,
    plt,
    release,
    sample_dropdown,
    savgol_filter,
    smooth_slider,
    stretch_slider,
    sustain,
    volume_slider,
):
    # --- Main Processing (Reactive to UI) ---
    row_idx = sample_dropdown.value
    row = df.row(row_idx, named=True)
    sample_id = row['Sample_ID']

    # Reactive Processing Block
    raw_values = np.array([v for k, v in row.items() if k != 'Sample_ID'], dtype=float)
    wavelengths = np.array([float(k.replace('X', '')) for k in row.keys() if k != 'Sample_ID'])

    # Audio Params
    duration, sr = 3.0, 44100

    # 1. Processing
    smoothed = savgol_filter(raw_values, window_length=smooth_slider.value, polyorder=3)
    base_cycle = librosa.util.normalize(smoothed - np.mean(smoothed))

    # 2. Pitch Shift / Resample
    new_len = int(len(base_cycle) * stretch_slider.value)
    resampled = librosa.resample(np.tile(base_cycle, 3), orig_sr=len(base_cycle), target_sr=new_len)
    audio_cycle = resampled[new_len : 2*new_len]

    # 3. Build Full Signal
    num_samples = int(sr * duration)
    full_signal = np.tile(audio_cycle, int(np.ceil(num_samples / len(audio_cycle))))[:num_samples]
    envelope = create_adsr_envelope(
            duration, sr, attack.value, decay.value, sustain.value, release.value
        )
    final_audio = full_signal * envelope * volume_slider.value

    # --- Visuals ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(wavelengths, smoothed, color='saddlebrown')
    ax[0].set_title("Coffee Spectrum (The Source)")

    ax[1].plot(envelope, color='red')
    ax[1].set_title("ADSR Envelope (Applied)")

    S = librosa.amplitude_to_db(np.abs(librosa.stft(final_audio)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax[2], cmap='magma')
    ax[2].set_title("Harmonic Fingerprint (The Sound)")
    plt.tight_layout()

    mo.vstack([
        mo.audio(final_audio, sr),
        fig
    ])
    return


if __name__ == "__main__":
    app.run()

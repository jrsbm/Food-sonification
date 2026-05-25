import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    app = mo.App(width="medium")
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import soundfile as sf
    from scipy.signal import savgol_filter
    from scipy.signal import find_peaks
    from midiutil import MIDIFile


@app.cell(hide_code=True)
def _():
    mo.md("""
    <div style="background: linear-gradient(135deg, #FFF8F0 0%, #FFF3E0 100%); padding: 48px 40px; border-radius: 12px; margin: 0 0 32px 0;">
    <h1 style="font-size: 2.8rem; font-weight: 900; margin: 0 0 8px 0; color: #8B4513; letter-spacing: -0.5px;">🎼 Food Fraud Ochestra</h1>
    <p style="font-size: 1.1rem; color: #A0522D; margin: 0; font-weight: 500;">Data Sonification Workshop</p>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 40px;">
    <div style="background: #E8F4F8; padding: 24px; border-radius: 8px; border-left: 4px solid #00CED1;">
    <h3 style="margin: 0 0 12px 0; color: #0077BE; font-size: 1.15rem; font-weight: 700;">🔊 Basic Synthesis</h3>
    <p style="margin: 0; color: #555; font-size: 0.95rem;">Sine waves, harmonics, sound foundations</p>
    </div>
    <div style="background: #F0E8F8; padding: 24px; border-radius: 8px; border-left: 4px solid #DA70D6;">
    <h3 style="margin: 0 0 12px 0; color: #8B008B; font-size: 1.15rem; font-weight: 700;">🎛️ Modulation</h3>
    <p style="margin: 0; color: #555; font-size: 0.95rem;">Frequency & amplitude control techniques</p>
    </div>
    <div style="background: #F8F0E8; padding: 24px; border-radius: 8px; border-left: 4px solid #FF8C00;">
    <h3 style="margin: 0 0 12px 0; color: #D2691E; font-size: 1.15rem; font-weight: 700;">☕ Data → Sound</h3>
    <p style="margin: 0; color: #555; font-size: 0.95rem;">Converting spectra into musical performance</p>
    </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _():
    freq_slider = mo.ui.slider(start=220, stop=880, step=20, value=440, label="Frequency (Hz)")
    duration_slider = mo.ui.slider(start=0.5, stop=3.0, step=0.5, value=1.0, label="Duration (s)")

    mo.md(f"""
    <div style="background: linear-gradient(135deg, #E8F4F8 0%, #F0F8FF 100%); padding: 36px 32px; border-radius: 12px; margin-bottom: 28px; border-top: 4px solid #00CED1;">
    <h2 style="margin: 0 0 24px 0; font-size: 2rem; font-weight: 800; color: #0077BE;">🔊 Sine Wave Synthesis</h2>
    <p style="margin: 0 0 24px 0; color: #333; font-size: 1.05rem;">Adjust frequency and duration to explore pure tone synthesis:</p>
    </div>
    """)
    return duration_slider, freq_slider


@app.cell(hide_code=True)
def _(duration_slider, freq_slider):
    # Styled slider container
    controls = mo.vstack([
        mo.md(f"<div style='background: #FFFACD; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;'><strong style='color: #FF8C00; font-size: 1.1rem;'>Frequency: {freq_slider.value} Hz</strong></div>"),
        freq_slider,
        mo.md(f"<div style='background: #FFE4E1; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;'><strong style='color: #DC143C; font-size: 1.1rem;'>Duration: {duration_slider.value}s</strong></div>"),
        duration_slider
    ])

    controls
    return


@app.cell(hide_code=True)
def _(duration_slider, freq_slider):
    # Audio setup
    f = freq_slider.value
    d = duration_slider.value
    sr_fm = 44100

    # 1. Generate the raw sine wave array
    t = np.linspace(0, d, int(sr_fm * d), endpoint=False)
    sine_wave = np.sin(2 * np.pi * f * t)

    # 2. Plot using Librosa's specialized 'waveshow'
    figfm, ax_fm = plt.subplots(figsize=(10, 3.5))

    # We zoom into the first 25ms so you can see the actual cycles, 
    # otherwise a 1-second wave just looks like a solid block of color.
    zoom_samples = int(sr_fm * 0.025) 
    librosa.display.waveshow(sine_wave[:zoom_samples], sr=sr_fm, ax=ax_fm, color="#00CED1")

    ax_fm.set_title(f"Sine Waveform: {f}Hz (Zoomed to first 25ms)", fontsize=13, fontweight='bold', color='#0077BE', pad=16)
    ax_fm.set_ylabel("Amplitude", fontsize=11, fontweight='600', color='#555')
    ax_fm.grid(True, alpha=0.2, linestyle='--', color='#0077BE')
    figfm.patch.set_facecolor('#F0F8FF')
    plt.tight_layout()

    # 3. Combine the audio player and the plot into the UI
    mo.vstack([
        mo.md("<div style='background: #E8F4F8; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;'><strong style='color: #0077BE; font-size: 1rem;'>▶ Listen to the waveform below:</strong></div>"),
        mo.center(mo.audio(sine_wave, rate=sr_fm)),
        figfm
    ])
    return


@app.cell(hide_code=True)
def _():
    ## Data import

    df1 = pl.read_csv('Instant_Coffee_Test_Samples.csv')
    sample_options = {f"ID: {df1[i, 'Sample_ID']} (Row {i})": i 
                      for i in range(len(df1))} 
    df2 = pl.read_csv('Dataset spices PTR-MS.csv')
    dfT = df2.transpose(include_header=True, header_name="mz_ratio")
    df_numeric = dfT.with_columns(
        pl.col("mz_ratio")
        .str.replace_all(r"[^\d.]", "")
        .cast(pl.Float64, strict=False)
    )
    sample_cols = [c for c in df_numeric.columns if c != "mz_ratio"]
    sample_names = {col: df_numeric[col][0] for col in sample_cols}
    df_plot = df_numeric.filter(pl.col("mz_ratio").is_not_null())
    df_plot = df_plot.with_columns([pl.col(c).cast(pl.Float64) for c in sample_cols])
    df_plot = df_plot.rename(sample_names)
    df_plot

    df_coffee = pl.read_csv('Dataset Arabica and Robusta coffee BARDS curves.csv')
    df_coffee
    return df1, df_coffee, df_plot, sample_options


@app.function(hide_code=True)
## Plot functions
def plot_spice_spectra(ax, df_plot, selected_samples=None):
    # Extract the X-axis (the mass-to-charge ratios)
    x_data = df_plot["mz_ratio"].to_numpy()

    # Define vibrant colors for spices
    colors = ["#8B008B", "#00CED1", "#DC143C", "#FF8C00", "#2E8B57", "#1E90FF", "#FFD700", "#FF1493"]

    # Loop through each column (each column is an individual spice sample) and plot it
    if not selected_samples:
        selected_samples = [c for c in df_plot.columns if c != "mz_ratio"]

    for idx, col in enumerate(selected_samples):
        if col in df_plot.columns:
            y_data = df_plot[col].to_numpy()
            color = colors[idx % len(colors)]
            ax.plot(x_data, y_data, label=col, alpha=0.8, lw=2, color=color)

    # Style the chart for Mass Spec data
    ax.set_title("PTR-MS Spice Fingerprints (Mass Spectra)", fontsize=14, fontweight='bold', color='#8B008B', pad=12)
    ax.set_xlabel("Mass-to-Charge Ratio (m/z)", fontsize=12, fontweight='600', color='#555')
    ax.set_ylabel("Intensity / Abundance", fontsize=12, fontweight='600', color='#555')
    ax.grid(True, linestyle=":", alpha=0.3, which="both", color='#999')
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor='#999')
    ax.set_facecolor('#F8E8F4')

    plt.tight_layout()


@app.cell(hide_code=True)
def _(sample_options):
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
    return (
        adsr_group,
        attack,
        decay,
        release,
        sample_dropdown,
        smooth_slider,
        stretch_slider,
        sustain,
        volume_slider,
    )


@app.cell(hide_code=True)
def _(
    adsr_group,
    sample_dropdown,
    smooth_slider,
    stretch_slider,
    volume_slider,
):
    # Layout the UI
    ui = mo.vstack([
        mo.md("""
        <div style="background: linear-gradient(135deg, #F8F0E8 0%, #FFF8F0 100%); padding: 36px 32px; border-radius: 12px; margin-bottom: 28px; border-top: 4px solid #FF8C00;">
        <h2 style="margin: 0 0 8px 0; font-size: 2rem; font-weight: 800; color: #D2691E;">☕ Coffee Sonification</h2>
        <p style="margin: 0; color: #8B4513; font-size: 1rem;">Transform coffee spectra into sound through interactive synthesis controls</p>
        </div>
        """),
        mo.md(f"""
        <div style="background: #FFF8DC; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; border-left: 3px solid #FF8C00;">
        <strong style="color: #D2691E; font-size: 1rem;">📊 Sample Selection:</strong>
        </div>
        """),
        sample_dropdown,
        mo.md("""
        <div style="background: #FFE4E1; padding: 16px 20px; border-radius: 8px; margin: 24px 0 20px 0; border-left: 3px solid #DC143C;">
        <strong style="color: #DC143C; font-size: 1rem;">🎛️ Synthesis Controls:</strong>
        </div>
        """),
        mo.hstack([
            mo.vstack([
                mo.md(f"<div style='background: #FFE4E1; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;'><span style='color: #DC143C; font-weight: 600;'>Pitch: {stretch_slider.value:.2f}x</span></div>"),
                stretch_slider
            ]),
            mo.vstack([
                mo.md(f"<div style='background: #FFDAB9; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;'><span style='color: #FF7F50; font-weight: 600;'>Smoothness: {smooth_slider.value}</span></div>"),
                smooth_slider
            ]),
            mo.vstack([
                mo.md(f"<div style='background: #FFD700; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;'><span style='color: #FF8C00; font-weight: 600;'>Volume: {volume_slider.value:.1f}</span></div>"),
                volume_slider
            ])
        ], justify="start", gap=3),
        mo.md("""
        <div style="background: #E8F4F8; padding: 16px 20px; border-radius: 8px; margin: 24px 0 20px 0; border-left: 3px solid #00CED1;">
        <strong style="color: #0077BE; font-size: 1rem;">📈 ADSR Envelope (Attack • Decay • Sustain • Release):</strong>
        </div>
        """),
        adsr_group
    ]).style({"padding": "0"})

    ui
    return


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    attack,
    decay,
    df1,
    release,
    sample_dropdown,
    smooth_slider,
    stretch_slider,
    sustain,
    volume_slider,
):
    # --- Main Processing (Reactive to UI) ---
    row_idx = sample_dropdown.value
    row = df1.row(row_idx, named=True)
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

    # Source spectrum
    ax[0].fill_between(wavelengths, smoothed, alpha=0.3, color='#8B4513')
    ax[0].plot(wavelengths, smoothed, color='#8B4513', linewidth=2.5)
    ax[0].set_title("☕ Coffee Spectrum (The Source)", fontsize=13, fontweight='bold', color='#8B4513', pad=12)
    ax[0].set_ylabel("Intensity", fontsize=11, fontweight='600', color='#555')
    ax[0].grid(True, alpha=0.2, linestyle='--')
    ax[0].set_facecolor('#FFF8F0')

    # ADSR envelope
    ax[1].fill_between(np.linspace(0, duration, len(envelope)), envelope, alpha=0.4, color='#DC143C')
    ax[1].plot(np.linspace(0, duration, len(envelope)), envelope, color='#DC143C', linewidth=2.5)
    ax[1].set_title("📈 ADSR Envelope (Applied)", fontsize=13, fontweight='bold', color='#DC143C', pad=12)
    ax[1].set_ylabel("Amplitude", fontsize=11, fontweight='600', color='#555')
    ax[1].grid(True, alpha=0.2, linestyle='--')
    ax[1].set_facecolor('#FFE4E1')

    # Spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(final_audio)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax[2], cmap='viridis')
    ax[2].set_title("🎼 Harmonic Fingerprint (The Sound)", fontsize=13, fontweight='bold', color='#0077BE', pad=12)
    ax[2].set_facecolor('#F0F8FF')
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()

    mo.vstack([
        mo.md("<div style='background: #FFF8F0; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;'><strong style='color: #8B4513; font-size: 1rem;'>▶ Listen & Analyze:</strong></div>"),
        mo.center(mo.audio(final_audio, sr)),
        fig
    ])
    return


@app.cell(hide_code=True)
def _(df_plot):
    # Get a list of all available spice names from the dataframe columns
    available_spices = [c for c in df_plot.columns if c != "mz_ratio"]

    # 1. Create a multi-select widget for manual picking
    spice_selector = mo.ui.multiselect(
        options=available_spices,
        value=available_spices[:3], # Default to picking the first 3 spices
        label="Choose Spices to Plot:"
    )
    return (spice_selector,)


@app.cell(hide_code=True)
def _(spice_selector):
    # Display the UI elements stacked neatly
    mo.vstack([
        mo.md("""
        <div style="background: linear-gradient(135deg, #F8E8F4 0%, #F0E8F8 100%); padding: 36px 32px; border-radius: 12px; margin-bottom: 28px; border-top: 4px solid #DA70D6;">
        <h2 style="margin: 0 0 8px 0; font-size: 2rem; font-weight: 800; color: #8B008B;">🌶️ Spice Spectra Analysis</h2>
        <p style="margin: 0; color: #556B2F; font-size: 1rem;">Select spice samples to visualize their PTR-MS fingerprints</p>
        </div>
        """),
        mo.md(f"<div style='background: #F0E8F8; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;'><strong style='color: #8B008B; font-size: 1rem;'>📊 Plot Controls ({len(spice_selector.value)} selected):</strong></div>"),
        spice_selector
    ])
    return


@app.cell(hide_code=True)
def _(df_plot, spice_selector):
    fig_spice, ax_spice = plt.subplots(figsize=(12, 6))
    plot_spice_spectra(ax=ax_spice, df_plot=df_plot, selected_samples=spice_selector.value)

    # Style the figure
    ax_spice.set_title("Mass Spectrometry Fingerprints: Spice Analysis", fontsize=14, fontweight='bold', color='#8B008B', pad=16)
    ax_spice.set_xlabel("Mass-to-Charge Ratio (m/z)", fontsize=12, fontweight='600', color='#555')
    ax_spice.set_ylabel("Intensity / Abundance", fontsize=12, fontweight='600', color='#555')
    fig_spice.patch.set_facecolor('#F8E8F4')

    plt.tight_layout()
    fig_spice
    return


@app.cell(hide_code=True)
def _():
    def spice_to_midi(df_plot, sample_name):
        """
        Extracts peaks from a spice spectrum and converts them into an 8-bar MIDI track.
        Shifts the starting point to mz=26 and scales the timeline up to 160.
        """
        x_data = df_plot["mz_ratio"].to_numpy()
        y_data = df_plot[sample_name].to_numpy()

        peaks, _ = find_peaks(y_data, height=(20, 50000))

        midi = MIDIFile(1)
        track = 0
        time_start = 0
        midi.addTrackName(track, time_start, f"{sample_name} Spectrum")
        midi.addTempo(track, time_start, 120)  # 120 BPM default

        total_beats = 32.0
        mz_span = 160.0 - 26.0
        beats_per_mz = total_beats / mz_span

        note_data = []

        for p in peaks:
            mz_val = x_data[p]
            hz_val = y_data[p]

            note_time = (mz_val - 26.0) * beats_per_mz
            note_time = max(0.0, note_time)

            midi_note = int(round(librosa.hz_to_midi(hz_val)))
            midi_note = max(0, min(127, midi_note))

            midi.addNote(track, 0, midi_note, note_time, 0.5, 90)

            # Save a human-readable version for our preview table
            note_data.append({
                "m/z (Time)": round(mz_val, 2),
                "Bar": round((mz_val / 10) + 1, 1),
                "Hz (Pitch)": round(hz_val, 1),
                "MIDI Note": midi_note,
                "Musical Note": librosa.midi_to_note(midi_note)
            })

        # 4. Write the MIDI data to a bytes buffer
        import io
        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)
        midi_buffer.seek(0)

        return midi_buffer.getvalue(), note_data

    def spice_to_percussion_midi(df_plot, sample_name):
        """
        Extracts peaks from a spice spectrum and scales intensities linearly 
        to fit precisely within a percussion VST range: B2 (lowest) to D#6 (highest).
        """
        x_data = df_plot["mz_ratio"].to_numpy()
        y_data = df_plot[sample_name].to_numpy()

        peaks, _ = find_peaks(y_data, height=(20, 50000))

        midi = MIDIFile(1)
        track = 0
        time_start = 0
        midi.addTrackName(track, time_start, f"{sample_name} Percussion")
        midi.addTempo(track, time_start, 120)

        total_beats = 32.0
        mz_span = 160.0 - 26.0
        beats_per_mz = total_beats / mz_span

        perc_preview = []

        if len(peaks) == 0:
            import io
            buf = io.BytesIO()
            midi.writeFile(buf)
            return buf.getvalue(), perc_preview

        # Define target MIDI range limits (B2 to D#6)
        note_min = int(librosa.note_to_midi('B2'))  # 47
        note_max = int(librosa.note_to_midi('D#6')) # 87

        # Get the local min and max intensity values of the detected peaks
        peak_heights = y_data[peaks]
        min_y = peak_heights.min()
        max_y = peak_heights.max()

        for p in peaks:
            mz_val = x_data[p]
            hz_val = y_data[p]

            note_time = (mz_val - 26.0) * beats_per_mz
            note_time = max(0.0, note_time)

            # Linear scaling formula: maps min peak to B2 and max peak to D#6
            if max_y != min_y:
                scaled_note = note_min + ((hz_val - min_y) / (max_y - min_y)) * (note_max - note_min)
                midi_note = int(round(scaled_note))
            else:
                midi_note = note_min # Fallback if all peaks are perfectly flat

            # Percussion hits sound cleaner as tight sixteenth notes (0.25 beats long)
            midi.addNote(track, 0, midi_note, note_time, 0.25, 100)

            perc_preview.append({
                "m/z (Time)": round(mz_val, 2),
                "Bar": round((mz_val / 10) + 1, 1),
                "Raw Intensity": round(hz_val, 1),
                "MIDI Note": midi_note,
                "Percussion Key": librosa.midi_to_note(midi_note)
            })

        import io
        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)
        midi_buffer.seek(0)

        return midi_buffer.getvalue(), perc_preview

    return spice_to_midi, spice_to_percussion_midi


@app.cell(hide_code=True)
def _(df_plot, spice_selector, spice_to_midi, spice_to_percussion_midi):
    if spice_selector.value:
        target_spice = spice_selector.value[0]

        # Run both engines behind the scenes
        synth_bytes, synth_preview = spice_to_midi(df_plot, target_spice)
        perc_bytes, perc_preview = spice_to_percussion_midi(df_plot, target_spice)

        # Create the Melodic Synth visual block
        synth_card = mo.vstack([
            mo.md("""
            <div style="background: linear-gradient(135deg, #E8F4F8 0%, #F0F8FF 100%); padding: 20px; border-radius: 10px; margin-bottom: 16px; border-left: 4px solid #00CED1;">
            <h3 style="margin: 0 0 8px 0; font-size: 1.2rem; font-weight: 800; color: #0077BE;">🎹 Melodic Synth</h3>
            <p style="margin: 0; font-size: 0.95rem; color: #333;">Raw frequency mapping—unbounded notes</p>
            </div>
            """),
            mo.download(
                data=synth_bytes,
                filename=f"{target_spice}_melodic.mid",
                label="📥 Download Melodic MIDI",
                mimetype="audio/midi"
            )
        ]).style({
            "padding": "0"
        })

        # Create the Percussion VST visual block
        perc_card = mo.vstack([
            mo.md("""
            <div style="background: linear-gradient(135deg, #FFE4E1 0%, #FFF0F5 100%); padding: 20px; border-radius: 10px; margin-bottom: 16px; border-left: 4px solid #DC143C;">
            <h3 style="margin: 0 0 8px 0; font-size: 1.2rem; font-weight: 800; color: #DC143C;">🥁 Percussion Sampler</h3>
            <p style="margin: 0; font-size: 0.95rem; color: #333;">Scaled to B2–D#6 percussion range</p>
            </div>
            """),
            mo.download(
                data=perc_bytes,
                filename=f"{target_spice}_percussion.mid",
                label="📥 Download Percussion MIDI",
                mimetype="audio/midi"
            )
        ]).style({
            "padding": "0"
        })

        # Build unified view layout with the data preview on the bottom
        unified_ui = mo.vstack([
            mo.md(f"""
            <div style="background: linear-gradient(135deg, #F8E8F4 0%, #FFF8F0 100%); padding: 32px 28px; border-radius: 12px; margin-bottom: 28px; border-top: 4px solid #DA70D6;">
            <h2 style="margin: 0 0 8px 0; font-size: 2rem; font-weight: 800; color: #8B008B;">🎼 MIDI Sonification Deck</h2>
            <p style="margin: 0 0 4px 0; color: #555; font-size: 1rem;"><strong>Sample:</strong> {target_spice}</p>
            <p style="margin: 0; color: #999; font-size: 0.95rem;">Extracted {len(perc_preview)} actionable peaks across 16 bars</p>
            </div>
            """),
            mo.hstack([synth_card, perc_card], justify="start", gap=2),
            mo.md("---"),
            mo.md("""
            <div style="background: #F0F8FF; padding: 16px 20px; border-radius: 8px; margin: 24px 0 16px 0; border-left: 3px solid #00CED1;">
            <strong style="color: #0077BE; font-size: 1rem;">📋 Live Mapping Preview (Percussion)</strong>
            </div>
            """),
            mo.ui.table(perc_preview)
        ])

        unified_ui
    else:
        mo.md("""
        <div style="background: #FFEBE9; padding: 24px; border-radius: 8px; text-align: center; border-left: 3px solid #D32F2F;">
        <p style="margin: 0; color: #B71C1C; font-size: 1rem;"><strong>⚠️ Please select at least one spice above to generate tracks</strong></p>
        </div>
        """)

    unified_ui
    return


@app.cell(hide_code=True)
def _(df_coffee):
    # Establish a dedicated figure and axes
    fig_bards, ax_bards = plt.subplots(figsize=(11, 5.5))

    # Draw the acoustic resonance curves
    ax_bards.plot(df_coffee["Time(s)"], df_coffee["Arabica_kHz"], label="Arabica Coffee", color="#8B4513", lw=2.8, linestyle='-', alpha=0.85)
    ax_bards.plot(df_coffee["Time(s)"], df_coffee["Robusta_kHz"], label="Robusta Coffee", color="#FF8C00", lw=2.8, linestyle='--', alpha=0.85)
    ax_bards.fill_between(df_coffee["Time(s)"], df_coffee["Arabica_kHz"], alpha=0.1, color="#8B4513")
    ax_bards.fill_between(df_coffee["Time(s)"], df_coffee["Robusta_kHz"], alpha=0.1, color="#FF8C00")

    # Format the plot for scientific publication style
    ax_bards.set_title("BARDS: Acoustic Resonance During Dissolution", fontsize=14, fontweight="bold", color="#8B4513", pad=16)
    ax_bards.set_xlabel("Time (seconds)", fontsize=12, fontweight="600", color="#555")
    ax_bards.set_ylabel("Resonance Frequency (kHz)", fontsize=12, fontweight="600", color="#555")
    ax_bards.grid(True, linestyle=":", alpha=0.3, color="#999", which="both")
    ax_bards.legend(loc="lower right", fontsize=11, framealpha=0.95, edgecolor='#999')
    ax_bards.set_facecolor('#FFF8F0')
    fig_bards.patch.set_facecolor('#FFFFFF')

    plt.tight_layout()

    # Expose the figure object at the end of the cell for marimo to render
    mo.vstack([
        mo.md("""
        <div style="background: linear-gradient(135deg, #FFF8F0 0%, #FFE8D6 100%); padding: 32px 28px; border-radius: 12px; margin-bottom: 20px; border-top: 4px solid #D2691E;">
        <h2 style="margin: 0 0 8px 0; font-size: 2rem; font-weight: 800; color: #8B4513;">🎵 BARDS Acoustic Fingerprint</h2>
        <p style="margin: 0; color: #A0522D; font-size: 1rem;">Comparing dissolution resonance profiles between coffee varieties</p>
        </div>
        """),
        fig_bards
    ])
    return


if __name__ == "__main__":
    app.run()

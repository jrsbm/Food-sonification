# Food Fraud Orchestra: Food Sonification Workshop

*Turning spectroscopic food authenticity data into music.*

This repository was used for a food sonification workshop 'Food Fraud Orchestra' during the 2026 Authentic Food Fraud Festival & Conference in Dublin, Ireland. The interactive session explores how to transform food data into sound using the `Librosa` digital signal processing library.

---

## Quick start

This project uses the `uv` package manager, which automatically installs the correct version of Python and all necessery libraries.

### Install uv

Open your computer's terminal (Terminal on Mac, Powersell on Windows) and paste:

* Mac / Linux:

  ```
  Bash

  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* Windows:

  ```
  PowerShell

  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

*Restart your terminal window after installation finishes.*

### Navigate to project folder

Download or clone this repository, open your terminal, and change your directory to the workshop folder:

```
Bash

cd uv-sound
```

### Launch the Notebook

To run the notebook as a polished presentation, run:

```
Bash

uv run marimo run workshop.py
```

To see the underlying code, run:

```
Bash

uv run marimo edit workshop.py
```

# Development environment setup

TinyTPU splits its tools between WSL2 and Windows (see AGENTS.md §5a). Pinned versions are in [`versions.env`](versions.env). Exact Python package versions are in `uv.lock`.

## WSL2 (Ubuntu): simulation, formal, docs

1. **OSS CAD Suite**
   - Download the `linux-x64` archive for the release in `OSS_CAD_SUITE_RELEASE` from the YosysHQ `oss-cad-suite-build` GitHub releases.
   - Extract it to `~/tools/oss-cad-suite`. The release is in the archive's `VERSION` file.
   - Don't add it to your shell profile. Never source its own `environment` script: that puts the suite's Python first on PATH.
2. **uv**
   - Install uv, at least `UV_MIN_VERSION`. Upgrade with `uv self update`.
3. **Python environment**, from the repo root:
   ```bash
   uv sync
   ```
   This installs Python 3.12 if needed and creates `.venv` with all dependency groups (`sim`, `docs`).
4. **Every new shell**, from the repo root:
   ```bash
   source tools/env.sh
   ```
   This puts only the suite's `bin/` first on PATH, and warns if the installed release doesn't match the pin.
5. **Check**:
   ```bash
   verilator --version        # should match OSS_CAD_SUITE_VERILATOR, not Ubuntu's /usr/bin 5.032
   yosys -m slang -p 'help read_slang' | head -1
   uv run python -c "import cocotb, pyslang, numpy; print('ok')"
   ```

## Windows: Vivado, board programming, host driver

1. **Vivado**
   - Install `VIVADO_VERSION` (ML Standard edition; the free tier covers the xc7a35t) with 7-series devices and the cable drivers.
   - Check that Hardware Manager detects the Basys3.
2. **FTDI latency timer**
   - Device Manager → Ports → the Basys3 USB Serial Port → Properties → Port Settings → Advanced → Latency Timer = **1 ms**.
3. **uv**
   - At least `UV_MIN_VERSION`. Run `uv self update` if older; `pyproject.toml` refuses older versions.
4. **Host-driver Python environment**
   - The virtual environment **must not** be the repo's `.venv`. That one is a Linux venv owned by WSL, and the two would overwrite each other.
   - Point uv at a Windows-local folder, then install only the runtime dependencies (`numpy`, `pyserial`). In PowerShell, from the repo folder:
     ```powershell
     $env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\tinytpu\venv"
     uv sync --no-default-groups
     ```
   - To make the variable permanent, set it once as a user environment variable (System Properties → Environment Variables).
5. **Repo path from Windows**
   - The repo lives in WSL at `\\wsl.localhost\Ubuntu\home\<user>\Projects\FPGA\TinyTPU`.
   - If Vivado has trouble with this UNC path, map a drive letter to it (task J0.7 confirms which works).

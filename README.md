# Coastal Hydrodynamics and Wave Energy Generation

**Companion Code Repository for the Wiley Textbook**

*Authors: Alireza Shadmani, Mohammad Reza Nikoo, Amir H. Gandomi*

---

## Overview

This repository contains Jupyter notebooks, Python scripts, and model data for the textbook case studies on **coastal hydrodynamics** and **wave energy converter (WEC) site selection and optimization**. The workflow covers the full modeling pipeline from raw environmental data through SWAN wave model simulation to multi-criteria site selection and genetic-algorithm-based array layout optimization.

### Two Case Study Regions

| Region | Location | Data Period | Key Features |
|--------|----------|-------------|--------------|
| **California Bay** | San Francisco Coast, USA | Jan-Feb 2024 | Deep-water swell, complex bathymetry, rotated SWAN grid |
| **Oman Coast** | Gulf of Oman / Arabian Sea | Apr-May 2021 | Wind-sea dominated, nested boundaries (W/S/E), calibration runs |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/                        # Jupyter notebooks (teaching materials)
│   ├── 01_preprocessing/             # Data inspection & visualization
│   │   └── preprocessing.ipynb
│   ├── 02_model_setup/               # SWAN model interface & configuration
│   │   └── swan_runner.ipynb
│   ├── 03_site_selection/            # Multi-criteria decision making
│   │   ├── hybrid_ahp_topsis_site_selection.ipynb  (standalone)
│   │   └── swan_coupled_ahp_topsis.ipynb           (SWAN-coupled)
│   ├── 04_optimization/              # Genetic algorithm optimization
│   │   ├── GA_array.ipynb                          (standalone)
│   │   └── swan_coupled_ga_array.ipynb             (SWAN-coupled)
│   ├── 05_postprocessing/            # Output visualization techniques
│   │   └── postprocessing_swan.ipynb
│   └── 06_model_validation/          # Statistical validation metrics
│       └── model_validation.ipynb
├── code/                             # Original Python scripts
├── california_bay.F/                 # California Bay SWAN model files & data
│   ├── INPUT                         # SWAN input file
│   ├── GEBCO.asc                     # Bathymetry (GEBCO ASCII grid)
│   ├── WAVE.nc                       # ERA5 wave reanalysis (NetCDF)
│   ├── WIND.nc                       # ERA5 wind reanalysis (NetCDF)
│   ├── wind.inp                      # Formatted wind input for SWAN
│   └── BW2.TPAR                      # Wave boundary conditions (TPAR format)
└── Oman/                             # Oman Coast SWAN model files & data
    ├── model.swn                     # SWAN input file
    ├── gebco.asc                     # Bathymetry
    ├── Wave-2021-April-May.nc        # ERA5 wave reanalysis
    ├── Wind-2021-April-May.nc        # ERA5 wind reanalysis
    ├── wind.inp                      # Formatted wind input for SWAN
    └── B*.TPAR                       # Boundary condition files (W/S/E sides)
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/alirezashadmani/Coastal-Hydrodynamics-and-Wave-Energy-Generation.git
cd Coastal-Hydrodynamics-and-Wave-Energy-Generation
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and work through the notebooks in order (01 through 06). Notebooks in `01_preprocessing`, `03_site_selection` (standalone), `04_optimization` (standalone), `05_postprocessing`, and `06_model_validation` can run without SWAN installed. The SWAN-coupled notebooks require a working SWAN installation (see below).

---

## Recommended Notebook Sequence

| Step | Notebook | SWAN Required? | Description |
|------|----------|:--------------:|-------------|
| 1 | `01_preprocessing/preprocessing.ipynb` | No | Inspect bathymetry, wave, and wind input data |
| 2 | `02_model_setup/swan_runner.ipynb` | No* | Learn the SWAN interface utilities |
| 3a | `03_site_selection/hybrid_ahp_topsis_site_selection.ipynb` | No | AHP-TOPSIS with tabular data |
| 3b | `03_site_selection/swan_coupled_ahp_topsis.ipynb` | **Yes** | AHP-TOPSIS with SWAN model output |
| 4a | `04_optimization/GA_array.ipynb` | No | GA optimization with uniform wave field |
| 4b | `04_optimization/swan_coupled_ga_array.ipynb` | **Yes** | GA optimization with SWAN-resolved field |
| 5 | `05_postprocessing/postprocessing_swan.ipynb` | No | Temporal, spatial, spectral visualization |
| 6 | `06_model_validation/model_validation.ipynb` | No | Compute Bias, MAE, RMSE, CC, NSE, KGE |

*\*The SWAN runner notebook defines utility functions and inspects input files but does not execute SWAN.*

---

## SWAN Model Installation

The [SWAN](https://swanmodel.sourceforge.io/) (Simulating WAves Nearshore) model is a third-generation spectral wave model. The SWAN-coupled notebooks (3b, 4b) require a compiled SWAN executable on your system PATH.

### Linux (Ubuntu/Debian)

```bash
# 1. Install build dependencies
sudo apt-get update
sudo apt-get install -y gfortran gcc make wget

# 2. Download SWAN source (check https://swanmodel.sourceforge.io for latest version)
wget https://sourceforge.net/projects/swanmodel/files/swan/41.45/swan4145.tar.gz
tar -xzf swan4145.tar.gz
cd swan4145

# 3. Configure for serial or parallel build
#    For serial:
cp macros.inc platform.pl .
make config
make ser

#    For OpenMP parallel (recommended):
make config
make omp

# 4. Install
sudo cp swan.exe /usr/local/bin/swan.exe
sudo cp swanrun /usr/local/bin/swanrun
sudo chmod +x /usr/local/bin/swanrun

# 5. Verify installation
swanrun -h
```

**Notes for Linux:**
- SWAN requires a Fortran compiler (gfortran recommended)
- For MPI parallel runs, install `libopenmpi-dev` and build with `make mpi`
- Edit `macros.inc` if you need custom compiler flags or paths

### macOS

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install gfortran via GCC
brew install gcc

# 3. Download and extract SWAN source
wget https://sourceforge.net/projects/swanmodel/files/swan/41.45/swan4145.tar.gz
tar -xzf swan4145.tar.gz
cd swan4145

# 4. Configure and build
#    Edit macros.inc: set F90 = gfortran-14 (or your version)
#    Check your gfortran version with: ls /opt/homebrew/bin/gfortran*
make config
make omp

# 5. Install to a location on your PATH
sudo cp swan.exe /usr/local/bin/swan.exe
sudo cp swanrun /usr/local/bin/swanrun
sudo chmod +x /usr/local/bin/swanrun

# 6. Verify
swanrun -h
```

**Notes for macOS:**
- Apple Silicon (M1/M2/M3/M4): Homebrew installs to `/opt/homebrew/bin/`. Ensure `gfortran` points to the correct version
- Intel Macs: Homebrew installs to `/usr/local/bin/`
- If you encounter linker errors, try: `export LDFLAGS="-L/opt/homebrew/lib"` and `export CPPFLAGS="-I/opt/homebrew/include"`

### Windows

**Option A: Windows Subsystem for Linux (WSL) — Recommended**

```powershell
# 1. Enable WSL (run in PowerShell as Administrator)
wsl --install -d Ubuntu

# 2. Open Ubuntu terminal, then follow the Linux instructions above
sudo apt-get update
sudo apt-get install -y gfortran gcc make wget
# ... (continue with Linux steps)
```

**Option B: Native Windows Build**

```
1. Install Intel oneAPI HPC Toolkit (includes ifort/ifx Fortran compiler)
   Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html

2. Open Intel oneAPI Command Prompt

3. Download and extract SWAN source code

4. Navigate to the SWAN directory:
   cd swan4145

5. Edit macros.inc:
   - Set F90 = ifort (or ifx for newer versions)
   - Set appropriate Windows flags

6. Build:
   make config
   make ser

7. Add the SWAN directory to your system PATH:
   - System Properties > Environment Variables > Path > Edit > Add the SWAN directory

8. Verify:
   swan.exe -h
```

**Option C: Pre-compiled Binary**

Pre-compiled SWAN binaries for Windows are occasionally available at:
https://swanmodel.sourceforge.io/download/download.htm

Check the download page for the latest available binary distribution.

**Notes for Windows:**
- WSL is the most reliable option and closely matches production Linux environments
- Native Windows builds require a Fortran compiler (Intel ifort/ifx recommended)
- The `swanrun` script is a Perl/Shell script; on native Windows, you may need to invoke `swan.exe` directly
- Ensure all input file line endings are Unix-style (LF, not CRLF) if using native Windows

### Verifying the Installation

After installation, verify SWAN is working:

```bash
# Check SWAN is on PATH
which swanrun      # Linux/macOS
where swanrun      # Windows

# Run a quick test (from any SWAN model directory)
cd california_bay.F
swanrun -input INPUT -omp 4
```

A successful run produces a `norm_end` file containing the message `"SWAN normal end"`.

---

## Data Sources

| Dataset | Source | Resolution | Variables |
|---------|--------|-----------|-----------|
| Bathymetry | [GEBCO 2023](https://www.gebco.net/) | ~450 m (0.004 deg) | Elevation (m) |
| Wave reanalysis | [ERA5](https://cds.climate.copernicus.eu/) | 0.25 deg, hourly | Hs, Tp, MWD |
| Wind reanalysis | [ERA5](https://cds.climate.copernicus.eu/) | 0.25 deg, hourly | u10, v10 |

---

## License

This repository accompanies a Wiley textbook. Please cite the textbook when using this material in academic work.

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.

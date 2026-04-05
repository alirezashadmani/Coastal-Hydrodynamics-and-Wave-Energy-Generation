// Pseudocode for Oman Coast Nested Grid Setup
START
  // 1. Define Coarse Grid Domain
  CREATE Coarse_Grid with spherical coordinates covering Northern Indian Ocean;
  LOAD Bathymetry from ETOPO2 or GEBCO for Coarse_Grid;
  
  // 2. Define Coarse Grid Boundary Conditions
  LOAD Time_Varying_Spectral_Data from WAVEWATCH III or ERA5 for offshore boundaries;
  LOAD Time_Varying_Wind_Field from ERA5 for entire domain;
  
  // 3. Run Coarse Grid Simulation (e.g., in SWAN)
  SWAN(Coarse_Grid, Boundary_Conditions, Wind_Field);
  
  // 4. Generate Nested Boundary Conditions
  GENERATE NESTOUT FILE from Coarse_Grid simulation for the nested area;
  
  // 5. Define Nested Grid Domain
  CREATE Nested_Grid with higher resolution (e.g., 25m) for specific hotspots like Masirah Island;
  LOAD High_Resolution_Bathymetry from local surveys for Nested_Grid;
  
  // 6. Run Nested Grid Simulation
  SWAN(Nested_Grid, Boundary_Conditions from NESTOUT, Wind_Field);
  
  // 7. Post-process and analyze results for energy potential;
STOP
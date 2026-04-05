// Pseudocode for California Bay Nested Grid Setup (SWAN)
START
  // 1. Define Coarse Grid Domain
  CREATE Coarse_Grid covering San Francisco Bay-Delta region;
  LOAD Bathymetry from local surveys;
  
  // 2. Define Coarse Grid Boundary Conditions
  LOAD Time_Varying_Wave_Parameters from CDIP buoy station 029 for offshore boundaries;
  
  // 3. Run Coarse Grid Simulation
  SWAN(Coarse_Grid, Boundary_Conditions);
  
  // 4. Generate Nested Boundary Conditions
  GENERATE NESTOUT FILE from Coarse_Grid simulation for the nested area;
  
  // 5. Define Nested Grid Domain
  CREATE Nested_Grid with higher resolution for South San Francisco Bay;
  LOAD High_Resolution_Bathymetry from local surveys for Nested_Grid;
  
  // 6. Define Physics and Forcing for Nested Grid
  SET Whitecapping = 'WES'; // Van der Westhuysen for wind-dominated system
  SET Bottom_Friction = 'COLLINS' with coefficient 0.02;
  LOAD Spatially_Varying_Wind_Field from HRRR atmospheric model;
  LOAD Water_Level_Data from NOAA Station 9414523;
  
  // 7. Run Nested Grid Simulation
  SWAN(Nested_Grid, Boundary_Conditions from NESTOUT, Wind_Field, Water_Level_Data);
  
  // 8. Post-process and analyze results for wave energy potential;
STOP
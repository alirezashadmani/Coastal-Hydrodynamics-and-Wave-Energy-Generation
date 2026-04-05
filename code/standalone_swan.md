// Pseudocode for Standalone SWAN Simulation
START
  // 1. Define Model Settings in a SWAN Control File (.swn)
  HEADING 'Project Title'
  PROJECT 'Project Number' 'Scenario Description'
  
  // 2. Define Grid and Bathymetry
  CGRID REGULAR 50 100 0.05 0.05 0 0 0 // Regular grid with 50x100 cells
  INPGRID BOT REGULAR 50 100 0.05 0.05 0 0 0 EXCVAL -999.00
  READINP BOTTOM 1.0 'bathymetry.grd' 4 0 FREE
  
  // 3. Define Physics and Source Terms
  GEN3 WAM3 // Third-generation physics package
  WIND 10.0 90.0 // Uniform wind with speed 10 m/s from 90 degrees
  FRICTION JONSWAP 0.067
  BREAKING BATTJES 0.73
  
  // 4. Define Boundary Conditions
  BOUN SIDE JONSWAP 3.3 PEAK DSPR DEG // Defines a JONSWAP spectral boundary
  BOUN SIDE E CCW CONSTANT PAR 1.402 5 115.22 27.48 // Constant parameters for the eastern boundary
  
  // 5. Define Time and Space
  MODE NONSTATIONARY TWOD
  SET DEPMIN 0.05 GRAV 9.81
  
  // 6. Define Output Locations and Variables
  TABLE 'output_points.txt' 0.1 // Outputs data to a table file
  BLOCK 'output_grid.txt' HEAD 'Significant Wave Height' HSIG
  
  // 7. Execute Simulation
  COMPUTE
  
  // 8. Stop
  STOP
END
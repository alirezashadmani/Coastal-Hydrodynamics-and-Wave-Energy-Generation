// Pseudocode for Post-Processing Visualization Workflow
FUNCTION Visualize_SWAN_Output(Output_File_Path, Plot_Type):
  // 1. Load Data
  IF Output_File_Path is a TABLE file (point output):
    LOAD Time_Series_Data, Metadata using specific reader;
    Data_Format = "Temporal";
  ELSE IF Output_File_Path is a BLOCK file (grid output):
    LOAD Gridded_Data_Dictionary, Metadata using specific reader;
    Data_Format = "Spatial";
  ELSE:
    RETURN "Error: Unsupported File Type";

  // 2. Plotting Logic
  IF Plot_Type == "Time Series" AND Data_Format == "Temporal":
    SELECT Variable (e.g., 'Hsig');
    PLOT (Time vs. Variable); 
    ANNOTATE (Title, Labels, Units from Metadata);
  ELSE IF Plot_Type == "Spatial Map" AND Data_Format == "Spatial":
    SELECT Scalar_Variable (e.g., 'Significant wave height');
    EXTRACT Grid_Coordinates (Xp, Yp);
    RESHAPE Scalar_Variable_Data into 2D Grid;
    CREATE Contour_Plot (Grid_Coordinates, Scalar_Variable_Data); // Heatmap
    OVERLAY Mean_Direction_Vectors ('Dir'); // Quiver plot showing wave direction
    ADD Colorbar;
  ELSE IF Plot_Type == "Spectral":
    LOAD Spectral_Density_Data (E(f, theta));
    CREATE Polar_Plot or Contour_Plot (Frequency vs. Direction);

  RETURN Plot_Object;
END FUNCTION
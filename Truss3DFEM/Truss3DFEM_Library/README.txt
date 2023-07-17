Instructions:

1) Install Anaconda
2) Create a new folder, e.g. "My_project_folder" somewhere on your PC/laptop.
3) Copy and paste the "Truss3DFEM_Library" folder and "Truss_3D_example.py" file in "My_project_folder"
4) From the Anaconda Navigator, launch PyCharm Community IDE, click File --> Open and select "My_project_folder" as the project folder. Click "OK"

5) Create a conda environment and set the project interpreter:
	- click on "add interpreter" --> select "conda environment"
	- set "C:\ProgramData\Anaconda3\envs\My_project_folder" as "location"
	- click "OK"
6) Go back to the Anaconda Navigator and make sure the "Applications on" dropdown menu is set to "My_project_folder"
7) Launch/install the conda Powershell Prompt
8) Type: 
	- "conda install -c conda-forge pyvista"
	- "conda install -c conda-forge matplotlib"
	- "conda install -c conda-forge prettytable"
	- "conda install -c conda-forge vtk=9.1.0"
9) Go back to PyCharm and run the "Truss_3D_example.py"
10) Enjoy




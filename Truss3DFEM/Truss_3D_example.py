from Truss3DFEM_Library.Truss3DFEMSolver import *

#-------------------------INPUTS -------------------------#
Truss_lenght = 20000. #[mm]
Truss_height = 1500. #[mm]
Truss_width = Truss_height
Cross_section_Area = 300. #[mm^2]
Elastic_modulus = 210000. #[N/mm^2]

#-------------------------Generate Node list & Connectivity list -------------------------#
par =  int((Truss_lenght / Truss_height))
bay_length = Truss_lenght / (par - 1)

my_node_list = []
for j in range(0, 2):
    for i in range(0, par):
        x = i * bay_length
        y = j * Truss_width
        z = Truss_height
        my_node_list.append([x, y, z])
for i in range(0, par):
    x = i * bay_length
    y = Truss_width/2.
    z = 0.
    my_node_list.append([x, y, z])

my_edge_list = []

for j in range(0, 3):
    for i in range(0, par-1):
        st_node = i + par * j
        end_node = i+1 + par * j
        my_edge_list.append([st_node, end_node])

for j in range(1, 3):
    for i in range(0, par):
        st_node = i
        end_node = (i + par * j)
        my_edge_list.append([st_node, end_node])
for i in range(par, 2 * par):
    st_node = i
    end_node = i + par
    my_edge_list.append([st_node, end_node])

for i in range(0, par-1):
    st_node = i
    end_node = i + par + 1
    my_edge_list.append([st_node, end_node])


for i in range(0, par-1):
    st_node = i
    end_node = i + 2 * par + 1
    my_edge_list.append([st_node, end_node])

for i in range(0, par-1):
    st_node = i + par
    end_node = i + 2 * par + 1
    my_edge_list.append([st_node, end_node])


#-----------------------Analyse Structural system---------------------------------#
my_structure = CreateSystem()
my_structure.AddGeometry(node_list=my_node_list, edge_list=my_edge_list)

my_structure.AddDOFRestrain(node_index=38, direction="all")
my_structure.AddDOFRestrain(node_index=12, direction="all")
my_structure.AddDOFRestrain(node_index=25, direction="all")

for i in range(0, len(my_node_list)):
    if i != 38:
        if i != 12:
            if i != 25:
                my_structure.AddPointLoad(node_index=i, direction="z", magnitude=-4000.)


my_structure.SetSectionsProperties(area_section=Cross_section_Area, elastic_mod=Elastic_modulus)

my_results = AnalyseSystem()

my_results.GetNodalDisplacements(my_structure, print_disp=True)
my_member_forces = my_results.GetMembersForces(my_structure, print_forces=True)

my_deformed_geometry = my_results.GetDeformedGeometry(my_structure)



#-------------------------Plot and print out geometry and results---------------#
my_geometry = GeomUtilities()

my_geometry.print_geom(node_list=my_node_list,
                       edge_list=my_edge_list
                       )

my_point_loads = my_structure.PointLoad_List

my_geometry.draw_geometry(node_list=my_node_list,
                          edge_list=my_edge_list,
                          show_solid_sections=True,  #optional param
                          section_radius=60,  #optional param
                          show_node_index=True,  #optional param
                          #show_elem_index=True, #optional param
                          deformed_geom=my_deformed_geometry,  #optional param
                          #show_member_forces=my_member_forces, #optional param
                          deform_magnitude=3.,  #optional param
                          point_loads=my_point_loads #optional param
                          )




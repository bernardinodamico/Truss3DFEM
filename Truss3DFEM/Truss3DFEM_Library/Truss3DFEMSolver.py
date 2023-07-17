import matplotlib.pyplot as plt
import numpy as np
import math
from prettytable import PrettyTable
import pyvista


class GeomUtilities():
    def scale_displacement_vec(self, rest_coord, displaced_coord, scale_factor):
        final_coord = rest_coord + (displaced_coord - rest_coord) * scale_factor
        return final_coord

    def get_deformed_geom(self, node_list, deformed_geom, deform_magnitude):
        node_list_deform = []
        for i in range(0, len(node_list)):
            x = self.scale_displacement_vec(node_list[i][0], deformed_geom[i][0], deform_magnitude)
            y = self.scale_displacement_vec(node_list[i][1], deformed_geom[i][1], deform_magnitude)
            z = self.scale_displacement_vec(node_list[i][2], deformed_geom[i][2], deform_magnitude)

            node = [x, y, z]
            node_list_deform.append(node)

        return node_list_deform

    def get_mid_nodes(self, node_list, edge_list):
        list_mid_nodes = []
        for i in range(0, len(edge_list)):
            st = edge_list[i][0]
            end = edge_list[i][1]

            st_x = node_list[st][0]
            st_y = node_list[st][1]
            st_z = node_list[st][2]

            end_x = node_list[end][0]
            end_y = node_list[end][1]
            end_z = node_list[end][2]

            x = (end_x + st_x) / 2.
            y = (end_y + st_y) / 2.
            z = (end_z + st_z) / 2.
            list_mid_nodes.append([x, y, z])

        return list_mid_nodes

    def P_load_arrow(self, p, point_loads, nodes):
        for p_load in point_loads:
            node_id = p_load[0]
            magnitude = p_load[2]

            if p_load[1] == "x" and magnitude < 0.:
                dir = [-1, 0, 0]
            elif p_load[1] == "x" and magnitude > 0.:
                dir = [1, 0, 0]
            elif p_load[1] == "y" and magnitude < 0:
                dir = [0, -1, 0]
            elif p_load[1] == "y" and magnitude > 0:
                dir = [0, 1, 0]
            elif p_load[1] == "z" and magnitude < 0:
                dir = [0, 0, -1]
            elif p_load[1] == "z" and magnitude > 0:
                dir = [0, 0, 1]

            p.add_arrows(cent=np.array(nodes[node_id]), direction=np.array(dir), mag=500., show_scalar_bar=False)

        return

    def draw_geometry(self,
                      node_list,
                      edge_list,
                      show_solid_sections=False,
                      section_radius=None,
                      show_node_index=False,
                      show_elem_index=False,
                      deformed_geom=None,
                      show_member_forces=None,
                      deform_magnitude= 1,
                      point_loads=None
                      ):
        """
        Note: the section_radius argument can be inputted as:
            - an singe float: in that case the same value is assigned to all elements
            - a list of floats: each element in the edge_list will be assigned the section_radius in the list based on the list index.
            If so, make sure the section_radius list has same length of the edge_list.
        """

        p = pyvista.Plotter()
        p.set_background("lightgrey", top="whitesmoke")
        p.add_axes(color="black")
        p.enable_trackball_style()
        p.add_camera_orientation_widget(animate=True, n_frames=20)

        if point_loads != None:
            if deformed_geom != None:
                list_def_nodes = self.get_deformed_geom(node_list, deformed_geom, deform_magnitude)
                self.P_load_arrow(p, point_loads, list_def_nodes)
            else:
                self.P_load_arrow(p, point_loads, node_list)

        if isinstance(section_radius, list) == False:
            section_radius = [section_radius] * len(edge_list)

        for edge in edge_list:
            st_index = edge[0]
            end_index = edge[1]

            nodes = np.array([node_list[st_index], node_list[end_index]])
            line = pyvista.Line(pointa=nodes[0], pointb=nodes[1])

            if deformed_geom != None:
                p.add_mesh(line, color=[0.650, 0.658, 0.670], line_width=0.7)
            else:
                if show_solid_sections != False:
                    if section_radius != None:
                        spline = pyvista.Spline(nodes).tube(radius=section_radius[edge_list.index(edge)])
                        p.add_mesh(spline, color=[0.470, 0.6, 0.788], smooth_shading=True)
                    else:
                        p.add_mesh(line, color=[0.027, 0.372, 0.874], line_width=1.7)
                else:
                    p.add_mesh(line, color=[0.027, 0.372, 0.874], line_width=1.7)


        if deformed_geom != None:
            p.add_text("deformation factor = "+str(deform_magnitude), position='upper_left', font_size=8, color="black", font="arial")
            list_def_nodes = self.get_deformed_geom(node_list, deformed_geom, deform_magnitude)

            for edge in edge_list:
                st_id = edge[0]
                end_id = edge[1]

                if show_solid_sections != False:
                    if section_radius != None:
                        nodes = np.array([list_def_nodes[st_id], list_def_nodes[end_id]])
                        spline = pyvista.Spline(nodes).tube(radius=section_radius[edge_list.index(edge)])
                        p.add_mesh(spline, color=[0.470, 0.6, 0.788], smooth_shading=True)
                    else:
                        line = pyvista.Line(pointa=np.array(list_def_nodes[st_id]), pointb=np.array(list_def_nodes[end_id]))
                        p.add_mesh(line, color=[0.027, 0.372, 0.874], line_width=1.7)
                else:
                    line = pyvista.Line(pointa=np.array(list_def_nodes[st_id]), pointb=np.array(list_def_nodes[end_id]))
                    p.add_mesh(line, color=[0.027, 0.372, 0.874], line_width=1.7)

        if show_node_index == True:
            if deformed_geom != None:
                list_def_nodes = self.get_deformed_geom(node_list, deformed_geom, deform_magnitude)
                poly = pyvista.PolyData(np.array(list_def_nodes))
            else:
                poly = pyvista.PolyData(np.array(node_list))

            poly["My Labels"] = np.array([f"{i}" for i in range(poly.n_points)])
            p.add_point_labels(poly, "My Labels", point_size=0.1, font_size=12, text_color="black", shape_color="whitesmoke")


        if show_elem_index == True:
            if deformed_geom != None:
                list_def_nodes = self.get_deformed_geom(node_list, deformed_geom, deform_magnitude)
                list_mid_nodes = self.get_mid_nodes(list_def_nodes, edge_list)
            else:
                list_mid_nodes = self.get_mid_nodes(node_list, edge_list)

            poly = pyvista.PolyData(np.array(list_mid_nodes))
            poly["My Labels"] = np.array([f"({i})" for i in range(poly.n_points)])
            p.add_point_labels(poly, "My Labels", point_size=0.1, font_size=12, text_color="black", shape_color="whitesmoke")

        if show_member_forces != None:
            if deformed_geom != None:
                list_def_nodes = self.get_deformed_geom(node_list, deformed_geom, deform_magnitude)
                list_mid_nodes = self.get_mid_nodes(list_def_nodes, edge_list)
            else:
                list_mid_nodes = self.get_mid_nodes(node_list, edge_list)

            poly = pyvista.PolyData(np.array(list_mid_nodes))
            poly["My Labels"] = np.array([f"{str(round(show_member_forces[i], 3))}" for i in range(len(show_member_forces))])
            p.add_point_labels(poly, "My Labels", point_size=0.1, font_size=12, text_color="black", shape_color="whitesmoke")


        p.show()


        return

    def export_wireframe_to_OBJ(self, node_list, edge_list, pathToObjFile):
        objFile = open(pathToObjFile, 'w')
        objFile.write("# hello!\n")
        objFile.write("\n")
        objFile.write("# vertices\n")
        for node in node_list:
            objFile.write("v " + str(float(node[0])) + " " + str(float(node[1])) + " " + str(float(node[2])) + "\n")
        objFile.write("\n")
        objFile.write("# edges\n")
        for edge in edge_list:
            objFile.write("l " + str(int(edge[0]) + 1) + " " + str(int(edge[1]) + 1) + "\n")
        objFile.close()
        return

    def print_geom(self, node_list, edge_list):
        nodes_table = PrettyTable()
        if node_list is not None:
            nodes_table.field_names = ["Node ID", "x coord.", "y coord.", "z coord."]
            for node in node_list:
                ID = node_list.index(node)
                nodes_table.add_row(
                    [str(ID), str(round(node[0], 3)), str(round(node[1], 3)), str(round(node[2], 3))])
            print(nodes_table.get_string(title="Node list"))

        conn_table = PrettyTable()
        if edge_list is not None:
            print("\n")
            conn_table.field_names = ["Element ID", "start Node ID", "end Node ID"]
            for edge in edge_list:
                ID = edge_list.index(edge)
                conn_table.add_row([str(ID), str(edge[0]), str(edge[1])])
            print(conn_table.get_string(title="Connectivity list"))

        return

class CreateSystem:
    def __init__(self):

        self.NodeList  = None
        self.EdgeList = None
        self.RestrainedDOF_List = []
        self.PointLoad_List = []
        self.SectionArea_List = []
        self.ElasticModulus_List = []

        return

    def AddGeometry(self, node_list, edge_list):
        self.NodeList = node_list
        self.EdgeList = edge_list

        return

    def AddDOFRestrain(self, node_index, direction):
        """
        Parameters:
            node_index: integer
            derection: one of the following strings: "x", "y", "z", "all".
        """

        restrain = [node_index, direction]
        self.RestrainedDOF_List.append(restrain)

        return

    def AddPointLoad(self, node_index, direction, magnitude):
        p_load = [node_index, direction, magnitude]
        self.PointLoad_List.append(p_load)

        return

    def SetSectionsProperties(self, area_section, elastic_mod):
        """
        area_section and elastic modulus can be inputted as:
            - an singe float: in that case the same value is assigned to all elements
            - a list of floats: each element in the edge_list will be assigned the area_section (and/or) elastic_modulus in the list based on the list index.
            Make sure the area_section list and/or elastic_mod list have same length of the edge_list.
        """
        if isinstance(area_section, list) == True:
            self.SectionArea_List = area_section
        else:
            self.SectionArea_List = [area_section] * len(self.EdgeList)

        if isinstance(elastic_mod, list):
            self.ElasticModulus_List = elastic_mod
        else:
            self.ElasticModulus_List = [elastic_mod] * len(self.EdgeList)

        return

class AnalyseSystem(CreateSystem):

    def __init__(self):
        CreateSystem.__init__(self)

        self.ElemLenghts_List = []
        self.ElemCosines_List = []
        self.ElemStiffMatrix_List = []
        self.GlobalStiffMatrix = None
        self.K_ff_subMatrix = None
        self.nod_disp_list = None
        self.NodalDisplacements_List = None
        self.DeformedGeometry = None

        return

    def calc_elem_lenghts(self, CreateSystem):
        list_len = []
        list_cos = []
        for i in range(0, len(CreateSystem.EdgeList)):
            str_ID = CreateSystem.EdgeList[i][0]
            end_ID = CreateSystem.EdgeList[i][1]

            str_x = CreateSystem.NodeList[str_ID][0]
            str_y = CreateSystem.NodeList[str_ID][1]
            str_z = CreateSystem.NodeList[str_ID][2]

            end_x = CreateSystem.NodeList[end_ID][0]
            end_y = CreateSystem.NodeList[end_ID][1]
            end_z = CreateSystem.NodeList[end_ID][2]

            lenght = math.sqrt(
                (end_x - str_x)**2 +
                (end_y - str_y) **2 +
                (end_z - str_z) **2
            )

            cos_x = (end_x - str_x) / lenght
            cos_y = (end_y - str_y) / lenght
            cos_z = (end_z - str_z) / lenght

            list_len.append(lenght)
            list_cos.append([cos_x, cos_y, cos_z])

        self.ElemLenghts_List = list_len
        self.ElemCosines_List = list_cos

        return


    def calc_elem_stiff_matrix(self, CreateSystem):
        self.calc_elem_lenghts(CreateSystem)

        list_len = self.ElemLenghts_List
        list_cos = self.ElemCosines_List
        list_A = CreateSystem.SectionArea_List
        list_E = CreateSystem.ElasticModulus_List

        list_matrices = []
        for i in range(0, len(CreateSystem.EdgeList)):
            c_x = list_cos[i][0]
            c_y = list_cos[i][1]
            c_z = list_cos[i][2]
            L = list_len[i]
            A = list_A[i]
            E = list_E[i]

            matrix_coeff = [
                [c_x**2,    c_x*c_y,   c_x*c_z, -(c_x**2), -c_x*c_y,  -c_x*c_z],
                [c_x*c_y,   c_y**2,    c_y*c_z, -c_x*c_y,  -(c_y**2), -c_y*c_z],
                [c_x*c_z,   c_y*c_z,   c_z**2,  -c_x*c_z,  -c_y*c_z,   -c_z**2],
                [-(c_x**2), -c_x*c_y,  -c_x*c_z, c_x**2,    c_x*c_y,   c_x*c_z],
                [-c_x*c_y,  -(c_y**2), -c_y*c_z, c_x*c_y,   c_y**2,    c_y*c_z],
                [-c_x*c_z,  -c_y*c_z,  -c_z**2,  c_x*c_z,   c_y*c_z,    c_z**2]
            ]

            elem_matrix = np.array(matrix_coeff) * ((E * A) / L)
            list_matrices.append(elem_matrix)

        self.ElemStiffMatrix_List = list_matrices

        return

    def calc_global_stiff_matrix(self, CreateSystem):
        size = len(CreateSystem.NodeList) * 3
        global_stiff_matrix = np.zeros(shape=[size, size])

        self.calc_elem_stiff_matrix(CreateSystem)

        for i in range(0, len(CreateSystem.EdgeList)):
            elem_matrix = self.ElemStiffMatrix_List[i]
            str_ID = CreateSystem.EdgeList[i][0]
            end_ID = CreateSystem.EdgeList[i][1]

            DOFs_elem = [int(str_ID * 3),
                         int(str_ID * 3 + 1),
                         int(str_ID * 3 + 2),
                         int(end_ID * 3),
                         int(end_ID * 3 + 1),
                         int(end_ID * 3 + 2)
                         ]

            for row in range(0, 6):
                for col in range(0, 6):
                    global_stiff_matrix[DOFs_elem[row], DOFs_elem[col]] = global_stiff_matrix[DOFs_elem[row], DOFs_elem[col]] + elem_matrix[row, col]

        self.GlobalStiffMatrix = global_stiff_matrix

        return

    def force_vector(self, CreateSystem):
        force_vec = [0.] * len(CreateSystem.NodeList) * 3

        for item in CreateSystem.RestrainedDOF_List:
            Node_ID = item[0]
            restrained_dir = item[1]

            if restrained_dir == "all":
                DOF = Node_ID * 3
                force_vec[DOF] = "R_"+str(Node_ID)+"_x"
                force_vec[DOF+1] = "R_"+str(Node_ID)+"_y"
                force_vec[DOF+2] = "R_"+str(Node_ID)+"_z"
            else:
                string = "R_"+str(Node_ID)+"_"+restrained_dir
                if restrained_dir == "x":
                    DOF = Node_ID * 3
                elif restrained_dir == "y":
                    DOF = Node_ID * 3 + 1
                elif restrained_dir == "z":
                    DOF = Node_ID * 3 + 2

                force_vec[DOF] = string

        for item in CreateSystem.PointLoad_List:
            Node_ID = item[0]
            load_dir = item[1]
            load_magnitude = item[2]

            if load_dir == "x":
                DOF = Node_ID * 3
            elif load_dir == "y":
                DOF = Node_ID * 3 + 1
            elif load_dir == "z":
                DOF = Node_ID * 3 + 2

            force_vec[DOF] = load_magnitude

        return force_vec

    def calc_disps(self, CreateSystem):
        force_vec = self.force_vector(CreateSystem)

        force_vec_ff = []
        for f in force_vec:
            if isinstance(f, str) == False:
                force_vec_ff.append(f)

        K_ff = []
        self.calc_global_stiff_matrix(CreateSystem)
        for i in range(0, len(force_vec)):
            if isinstance(force_vec[i], str) == False:
                row = []
                for j in range(0, len(force_vec)):
                    if isinstance(force_vec[j], str) == False:
                        row.append(self.GlobalStiffMatrix[i, j])
                K_ff.append(row)

        self.K_ff_subMatrix = np.array(K_ff)
        force_vec_ff = np.array(force_vec_ff)

        disp_vector = np.linalg.solve(self.K_ff_subMatrix, force_vec_ff)

        self.nod_disp_list = disp_vector.tolist()

        return self.nod_disp_list

    def GetNodalDisplacements(self, CreateSystem, print_disp=False):
        self.calc_disps(CreateSystem)
        force_vec = self.force_vector(CreateSystem)

        disp_vector = []
        Index = 0
        for i in range(0, len(force_vec)):
            if isinstance(force_vec[i], str) != True:
                disp_vector.append(self.nod_disp_list[Index])
                Index = Index + 1
            else:
                disp_vector.append(0.)

        disp_list = []
        for i in range(0, len(CreateSystem.NodeList)):
            DOF_x = i * 3
            DOF_y = i * 3 + 1
            DOF_z = i * 3 + 2

            disp_x = disp_vector[DOF_x]
            disp_y = disp_vector[DOF_y]
            disp_z = disp_vector[DOF_z]

            disp_list.append([disp_x, disp_y, disp_z])

        if print_disp == True:
            disp_table = PrettyTable()
            disp_table.field_names = ["Node_ID", "x disp.", "y disp.", "z disp."]
            for i in range(0, len(disp_list)):
                disp_table.add_row([str(i),
                                    str(round(disp_list[i][0], 3)),
                                    str(round(disp_list[i][1], 3)),
                                    str(round(disp_list[i][2], 3))
                                    ])

            print(disp_table.get_string(title="Nodal displacements"))

        self.NodalDisplacements_List = disp_list

        return self.NodalDisplacements_List

    def GetDeformedGeometry(self, CreateSystem):
        rest_geom = CreateSystem.NodeList
        disp_geom = self.GetNodalDisplacements(CreateSystem)

        deformed_geom = []
        for i in range(0, len(rest_geom)):
            x = rest_geom[i][0] + disp_geom[i][0]
            y = rest_geom[i][1] + disp_geom[i][1]
            z = rest_geom[i][2] + disp_geom[i][2]
            deformed_geom.append([x, y, z])

        self.DeformedGeometry = deformed_geom

        return self.DeformedGeometry


    def GetMembersForces(self, CreateSystem, print_forces= False):
        self.GetNodalDisplacements(CreateSystem)

        list_forces = []
        for i in range(0, len(CreateSystem.EdgeList)):
            E = CreateSystem.ElasticModulus_List[i]
            A = CreateSystem.SectionArea_List[i]
            L = self.ElemLenghts_List[i]

            c_x = self.ElemCosines_List[i][0]
            c_y = self.ElemCosines_List[i][1]
            c_z = self.ElemCosines_List[i][2]

            st_node = CreateSystem.EdgeList[i][0]
            end_node = CreateSystem.EdgeList[i][1]

            v_1 = self.NodalDisplacements_List[st_node][0]
            v_2 = self.NodalDisplacements_List[st_node][1]
            v_3 = self.NodalDisplacements_List[st_node][2]
            v_4 = self.NodalDisplacements_List[end_node][0]
            v_5 = self.NodalDisplacements_List[end_node][1]
            v_6 = self.NodalDisplacements_List[end_node][2]

            Axial_force = ((E*A)/L) * (c_x*(v_4 - v_1) + c_y*(v_5 - v_2) + c_z*(v_6 - v_3))
            list_forces.append(Axial_force)

        if print_forces == True:
            f_table = PrettyTable()
            f_table.field_names = ["Element_ID", "Axial force"]
            for i in range(0, len(list_forces)):
                f_table.add_row([str(i),
                                str(round(list_forces[i], 3)),
                                ])

            print(f_table.get_string(title="Members forces"))

        return list_forces





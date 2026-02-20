from copy import copy, deepcopy
from typing import Any, Union
import numpy as np


from ..spinw import SpinW
from .supercell_plotter import SupercellPlotter
from ..utils.linalg import cartesian2spherical
# from ..crystal import Atom

from vispy import app
from vispy.scene import SceneCanvas
import vispy.scene.visuals as visuals
from vispy.scene.widgets import ViewBox
from vispy.scene.cameras import TurntableCamera
from vispy.visuals.transforms import MatrixTransform, STTransform, ChainTransform
from vispy.visuals.filters import ShadingFilter, Alpha


# DEV NOTES
# [ ] Check out InstancedMech visual. It stores the same mesh for multiple objects.
#     InstancedMeshdef.__init__(self, *args, instance_positions, instance_transforms, instance_colors=None, **kwargs)
#     So objects can be at different positions and be rescaled and rotated, and have various colors.
#     Apparently can vasty speed up rendering.
 
class VispySupercellPlotter(SupercellPlotter):
    '''
    Vispy is a 3D, ray-tracing renderer.
    '''
    # DEV notes
    # Visuals are great, but have problems with attaching filters.
    # For this reason the objects are created directly from meshes,
    # or through helpers from which the meshes are extracted.
    _objects: list[visuals.Mesh]
    _descriptions: list[str]
    _shaders: list[ShadingFilter]

    def __init__(self, sws: SpinW, show: bool=True, plot_options: dict={}):
        super().__init__(sws=sws)

        self.logger.setLevel('INFO')

        # TODO move these setting to config file
        # self.atom_alpha = 0.8
        # self.spin_scale = 2
        # self.arrow_width = 0.1
        # self.arrow_head_size = 3

        self.plot_options = plot_options


        # Init vispy objects
        self.canvas = SceneCanvas(bgcolor='white', show=show)
        self.view = ViewBox()
        self.view.camera = TurntableCamera()
        self.canvas.central_widget.add_widget(self.view)


        self.light_dir = [0, 1.0, 0, 0]
        self.light_move_sensitivity = 0.3
        self.light_transform = deepcopy(self.view.camera.transform) 
        # For some reason the light direction has to be transformed by this `black-box` function
        self.bb_light_dir = self.light_transform.imap(self.light_dir) 
        self._using_headlight = False

        # Enable object picking
        self._previously_clicked_id = None
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.key_press.connect(self.on_key_press)

        # self.timer = app.Timer(interval=1./120)
        # self.timer.connect(self.update_frame)


        self._objects = list()
        self._descriptions = list()
        self._shaders = list()

        return
    
    def default_shading_filter(self):
        sf = ShadingFilter(shading='smooth', shininess=20, ambient_coefficient=(0,0,0,0))
        sf.light_dir = self.light_dir[:3]
        self._shaders.append(sf)

        return sf
    
    def deploy(self) -> 'SceneCanvas':
        self.attach_headlight()
        # self.enable_picking_objects()

        self.plot(self.plot_options)

        self.view.camera.center = self._structure_center
        self.view.camera.distance = 2*self._largest_distance / np.tan(np.radians(self.view.camera.fov)/2)
        self.logger.warning(f'Center = {self._structure_center}')
        self.logger.warning(f'Distance = {self._largest_distance}')
        
        # self.timer.start(0)
        self.canvas.app.run()

    
        return self.canvas
    
    def update_frame(self, event):
        time = self.timer.elapsed
        print(f'{time=}')
        for obj in self._objects:
            obj.transform = STTransform(translate=np.array([0.2,0,0])*np.sin(2*np.pi*time/10))*obj.transform
        
    
    ### LIGHT HANDLING
    def update_light_dir(self):
        if self._using_headlight:
            self.bb_light_dir = self.light_transform.imap(self.light_dir)
            self.view.scene.transform.changed.emit()
            # self.view.camera.distance += 1e-10  # dirty trick to call camera update
        else:
            for sf in self._shaders:
                sf.light_dir = self.light_dir[:3]

            self.canvas.update()

    
    def attach_headlight(self):
        self._using_headlight = True

        @self.view.scene.transform.changed.connect
        def on_transform_change(event):
            transform = self.view.camera.transform
            # print(f'DEBUG: initial {self.bb_light_dir} transformed {transform.map(self.bb_light_dir)[:3]}')
            for sf in self._shaders:
                sf.light_dir = transform.map(self.bb_light_dir)[:3]
    
    def on_key_press(self, event):
        if event.key == 'q':
            self.light_dir[0] += self.light_move_sensitivity
            self.update_light_dir()
        if event.key == 'a':
            self.light_dir[0] -= self.light_move_sensitivity
            self.update_light_dir()
        if event.key == 'w':
            self.light_dir[1] += self.light_move_sensitivity
            self.update_light_dir()
        if event.key == 's':
            self.light_dir[1] -= self.light_move_sensitivity
            self.update_light_dir()
        if event.key == 'e':
            self.light_dir[2] += self.light_move_sensitivity
            self.update_light_dir()
        if event.key == 'd':
            self.light_dir[2] -= self.light_move_sensitivity
            self.update_light_dir()

    def on_mouse_press(self, event):
        clicked_object = self.canvas.visual_at(event.pos)
        if clicked_object in self._objects:
            id = self._objects.index(clicked_object)
            print(f'DEBUG {id=}')
            print(self._descriptions[id])

    
    ### IMPLEMENTING ABSTRACT UFNCTIONS
    def plot_balls(self, 
                   positions: np.ndarray, 
                   sizes: np.ndarray, 
                   colors: np.ndarray):
        colors = colors/255.

        for position, size, color in zip(positions, sizes, colors):
            sphere = visuals.Sphere(radius=size, method='latitude', color=color)
            mesh = sphere.mesh.mesh_data
            obj = visuals.Mesh(vertices=mesh.get_vertices(),faces=mesh.get_faces(), color=color)
            
            obj.attach(self.default_shading_filter())

            obj.transform = STTransform(translate=position)
            obj.interactive = True

            self.view.add(obj)
            self._objects.append(obj)
            self._descriptions.append(f'Atom: {position=} {size=}')
        
        return
        
    def plot_labels(self,
                    positions: np.ndarray,
                    labels: np.ndarray,
                    colors: np.ndarray):
        
        colors = colors/255.
        # Labels
        # font size was nice and tricky in original vispy implementation
        obj = visuals.Text(pos=positions, 
                                 text=labels, 
                                 color="white", 
                                 font_size=12)
        obj.interactive = True
        
        self.view.add(obj)

        self._objects.append(obj)   # type: ignore
        self._descriptions.append(f'Label: text={labels}')
        return obj

    def _arrow_points(self, length, tail_width, head_length, head_width, 
                      overhang=0.05, head_Bezier_control=[[0.4], [0.8]],
                      cap_res=1, head_res=16) -> tuple[list, list]:
        points_cap = [[0,0, (1-np.cos(th))*tail_width/2] for th in np.linspace(0, np.pi/2, cap_res+1)]
        points_tail = [[0,0,length-head_length+overhang]]

        radii_cap = list(tail_width*np.sin(np.linspace(0, np.pi/2, cap_res+1)))
        radii_tail = [tail_width]

        # Bezier curve for the head, where P1 is the control point.
        # Here we are looking at the 
        # P0 - - - * P1
        # *  * * * \ *   
        # *  * * * * \   
        # *  * * * * |
        # 0  * * * * P2
        t = np.linspace(0,1,head_res)
        P0, P1, P2 = np.array([[0],[1]]), np.array(head_Bezier_control), np.array([[1],[0]])
        Bx, By = P1 + (1-t)**2*(P0-P1)+t**2*(P2-P1)
        # self.logger.info(f'Bx={Bx}')
        points_head = [[0,0,x] for x in length-head_length + Bx*head_length] # Map Bx (0,1) to (length-head_length, length)
        radii_head = list(By*head_width)
        # self.logger.info(f'{points_head}, {radii_head}')

        # points_head = list(np.linspace([0,0,length-head_length], [0,0,length], head_res))
        # radii_head = list(head_width*( np.sqrt(np.linspace(3,0,head_res)+1)-1 ))

        return (points_cap+points_tail+points_head, radii_cap+radii_tail+radii_head)
    
    def plot_arrows(self,
                    positions: np.ndarray,
                    directions: np.ndarray,
                    colors: np.ndarray):
        colors = colors/255.

        # Magnetic moments from magnetic atoms only
        # def plot_magnetic_structure(self, canvas_scene, mj, pos, colors):
        cap_res = 16
        tail_width = self.arrow_width
        head_width = self.arrow_head_size*tail_width

        cap_res = 1
        ang_res = 64

        # obj = visuals.XYZAxis(parent=self.canvas.scene)


        for position, direction, color in zip(positions, directions, colors):
            r, th, phi = cartesian2spherical(direction)
            points, radii = self._arrow_points(length=r*self.spin_scale, 
                                               tail_width=tail_width, 
                                               head_length=0.5, head_width=head_width, 
                                               overhang=-0.05, 
                                               cap_res=cap_res, head_res=16)
            a = visuals.Tube(points=points, radius=radii, tube_points=ang_res)
            mesh = a.mesh_data
            obj = visuals.Mesh(vertices=mesh.get_vertices(), faces=mesh.get_faces(), color=color)
            obj.attach(self.default_shading_filter())
            
            m1, m2 = MatrixTransform(), MatrixTransform()
            m1.rotate(np.degrees(th), (0,1,0))
            m2.rotate(np.degrees(phi), (0,0,1))
            m2.translate(position)
            obj.transform = ChainTransform([m2, m1])
            obj.interactive = True
            
            self.view.add(obj)
            self._objects.append(obj)
            self._descriptions.append(f'Arow: {position=} {direction=}')

        return
    
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray,
                   width: float=0.2,
                   alpha: float=1):
        '''
        width in px
        '''
        colors = colors/256
        ang_res = 32
        for line, color in zip(lines, colors):
            ### With tube
            # width = width/10
            # dir = np.array(line[1])-np.array(line[0])
            # dir /= np.linalg.norm(dir)
            # points = [line[0], line[0]+dir*width/2, line[1]-dir*width/2, line[1]]
            # l = visuals.Tube(points=points, radius=[1e-10,width,width,1e-10], tube_points=ang_res, shading='flat')

            ### With line
            obj = visuals.Line(line, width=width, color=color, antialias=True, parent=self.view)
            # obj.interactive = True
            
            self.view.add(obj)
            self._objects.append(obj)
            self._descriptions.append(f'Line from={line[0]} to={line[1]} {color=}')

    def plot_ellipsoids2(self, positions, matrices, colors):
        colors = colors/255.

        for position, matrix, color in zip(positions, matrices, colors):
            sphere = visuals.Sphere(radius=1, method='latitude', color=color)


            mesh = sphere._mesh.mesh_data
            obj = visuals.Mesh(vertices=mesh.get_vertices(),
                                faces=mesh.get_faces(), color=color)
            
            obj.transform = self._MTfromMatrix(position, matrix)
            
            obj.attach(self.default_shading_filter())
            obj.attach(Alpha(0.5))
            obj.interactive = True
            
            self.view.add(obj)
            self._objects.append(obj)
            self._descriptions.append(f'Ellipsoid: {obj.transform=}')

    def _MTfromMatrix(self, position: np.ndarray, matrix: np.ndarray) -> MatrixTransform:
        '''Determine the `MatrixTransform` from the `matrix`
        that describes an ellipsoid.'''
        # See Wiki to get the transform
        # https://en.wikipedia.org/wiki/Ellipsoid#Parametric_representation
        e_vals, e_vecs = np.linalg.eigh(matrix)
        M = np.zeros((4,4))
        # With the vispy convention this matrix seems like v' = v M
        # or whatever. It seems to work.
        M[:3,0] = e_vals[0]*e_vecs[0] + 1e-10
        M[:3,1] = e_vals[1]*e_vecs[1] + 1e-10
        M[:3,2] = e_vals[2]*e_vecs[2] + 1e-10
        M[3,:3] = position
        M[3,3] = 1
        # self.logger.info(f'Plotting ellipsoid with matrix \n {M}')
        # self.logger.info(f'Ellipsoid evals = {e_vals}')
        # self.logger.info(f'Ellipsoid evecs = {e_vecs}')

        return MatrixTransform(matrix=M)

    def plot_ellipsoids(self, positions, matrices, colors):
        '''Plots a flat ellipse/sphere at `positions` with main axes orientation from `matrices`.
        Assumes one of the eigenvalues in each `matrices` is zero.
        '''
        colors = colors/255.
        for position, matrix, color in zip(positions, matrices, colors):
            obj = visuals.Ellipse(center=(0,0,0), radius=(1,1), 
                                  color=color, 
                                  border_color=np.clip(0.5*color, 0,1), border_width=2)

            obj.transform = self._MTfromMatrix(position, matrix)
            obj.attach(Alpha(0.5))
            obj.interactive = True
            
            self.view.add(obj)
            self._objects.append(obj)
            self._descriptions.append(f'Ellipsoid: {obj.transform=}')




            
    def present_arrows(self):
        '''Show types of arrows that can be used.'''
        # Fundamental arrow proerties are position and direction.
        # Then, we have all styling properties:
        # width: width of the tail
        # head_width: width of the head at its base
        # head_length
        # overhang: proportion at which the arrow is swept back [-1,1]. 0 for Triangular arrow
        # tail_cap

        cap_res = 16
        tail_width = 0.1
        head_width = 3*tail_width

        cap_res = 1
        ang_res = 64

        obj = visuals.XYZAxis(parent=self.canvas.scene)

        def arrow_points(length, tail_width, head_length, head_width, overhang, cap_res) -> tuple[list, list]:
            points_cap = [[0,0, (1-np.cos(th))*tail_width/2] for th in np.linspace(0, np.pi/2, cap_res+1)]
            points_tail_head = [[0,0,length-head_length-1e-5],[0,0,length-head_length],[0,0,length]]

            radii_cap = list(tail_width*np.sin(np.linspace(0, np.pi/2, cap_res+1)))
            radii_tail_head = [tail_width, head_width, 1e-10]

            return (points_cap+points_tail_head, radii_cap+radii_tail_head)

        ###
        for nx in range(0,10+1,2):
            for ny in range(0,10+1, 2):
                B = np.array([[nx/10],[ny/10]])
                points, radii = self._arrow_points(length=2, tail_width=0.2, 
                                             head_length=0.6, head_width=0.5, 
                                             overhang=-0.02, cap_res=cap_res, head_res=8,
                                             head_Bezier_control=B)
                a1 = visuals.Tube(points=points, radius=radii, tube_points=ang_res, shading='flat')
                
                a1.transform = STTransform(translate=[nx,ny,0])
                l = visuals.Text(f'Bx={B[0]} By={B[1]}', pos=[nx, ny, 2.2], font_size=40)
                self.view.add(a1)
                self.view.add(l)
from copy import deepcopy
import logging
import traceback
import numpy as np

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..spinw import SpinW
    from ..crystal import Atom

from ..utils.linalg import cartesian2spherical, RfromZ

from vispy.scene import SceneCanvas
from vispy.scene.widgets import ViewBox
from vispy.scene.cameras import TurntableCamera
import vispy.scene.visuals as visuals
from vispy.visuals.transforms import MatrixTransform, STTransform
from vispy.visuals.filters import ShadingFilter, Alpha, TextureFilter
from vispy.util.event import EmitterGroup, Event
from vispy.app import Timer



# DEV NOTES
# [ ] Check out InstancedMech visual. It stores the same mesh for multiple objects.
# InstancedMeshdef.__init__(self, *args, instance_positions, instance_transforms, instance_colors=None, **kwargs)
# So objects can be at different positions and be rescaled and rotated, and have various colors.
# Apparently can vasty speed up rendering.
# Objects that can share a mesh: [ALL atoms, arrows depicting moments of equal length, unit cell edges]
#
# [ ] Implement objects picking. For that I might need to add descriptions to instancedmesh.


class OptionsManager(object):
    '''Interface for the plotter options. 
    
    - Holds default values.
    - Validates variable types, values, and relations between variables values.
    - Emits appropriate signals on change.

    Notes
    -----
    To add a new option:
    1. Add entry in __attribute_defaults.
    2. Add validator in self.__setattr__().
    3. Connect to a signal in self__init__().
    4. Add property in self.
    '''
    # Use atomic types
    __attribute_defaults = [
        ('light_dir_x', 0.0),
        ('light_dir_y', 1.0),
        ('light_dir_z', 0.0),
        ('light_dir_w', 0.0),
        ('light_move_step', 0.2),
        ('light_headlight_on', True),
        ('boundaries_v1_x', 1.0),
        ('boundaries_v1_y', 0.0),
        ('boundaries_v1_z', 0.0),
        ('boundaries_v1_lim_1', 0.0),
        ('boundaries_v1_lim_2', 1.0),
        ('boundaries_v2_x', 0.0),
        ('boundaries_v2_y', 1.0),
        ('boundaries_v2_z', 0.0),
        ('boundaries_v2_lim_1', 0.0),
        ('boundaries_v2_lim_2', 1.0),
        ('boundaries_v3_x', 0.0),
        ('boundaries_v3_y', 0.0),
        ('boundaries_v3_z', 1.0),
        ('boundaries_v3_lim_1', 0.0),
        ('boundaries_v3_lim_2', 1.0),
        ('scale_atom_radius', 1.0),
        ('scale_moment_length', 1.0),
        ('coordinates_xyz_scale', 33),
        ('coordinates_xyz_position', 3),
        ('coordinates_xyz_offset_x', -88),
        ('coordinates_xyz_offset_y', -66),
        ('coordinates_uvw_scale', 11),
        ('coordinates_uvw_position', 4),
        ('coordinates_uvw_offset_x', 88),
        ('coordinates_uvw_offset_y', -66),
    ]
    def __init__(self, options: dict[str, Any]={}):
        # Logger
        self.__setattr__('_logger', logging.getLogger('OptionsManager'), validate=False)

        for name, value in self.__attribute_defaults:
            self.__dict__[name] = value

        self._validate_boundaries_limits(self.__dict__)
        self._validate_boundaries_span(self.__dict__)

        self.logger.warning(f"Initiating `{self.__class__.__name__}` with: {options}")
        for name, value in options.items():
            self.__setattr__(name, value)


        # Create events list
        emitters = EmitterGroup(source=self, 
                                boundaries=Event, light_dir=Event, headlight=Event,
                                scale=Event, coordinates=Event)
        self.__setattr__('events', emitters, validate=False)
        
        self.events.connect(self._event_debugger)
        
    def _event_debugger(self, event):
        print('OPTIONS: Event triggered')
        print(event.__repr__())


    ### VALIDATORS
    def _validate_boundaries_limits(self, options: dict):
        '''Ensure that boundaries are in ascending order.'''
        assert options['boundaries_v1_lim_1'] < options['boundaries_v1_lim_2'], "Boundaries v1 must be in ascending order."
        assert options['boundaries_v2_lim_1'] < options['boundaries_v2_lim_2'], "Boundaries v2 must be in ascending order."
        assert options['boundaries_v3_lim_1'] < options['boundaries_v3_lim_2'], "Boundaries v3 must be in ascending order."

    def _validate_boundaries_span(self, options: dict):
        '''Ensure the vectors span 3D space.'''
        V = [[options.get(f'boundaries_v{vn}_{ri}') for ri in 'xyz'] for vn in [1,2,3]]
        assert np.linalg.matrix_rank(V) == 3, "Vectors definig boundaries must span 3D space."

    def _validate_scales(self, options: dict):
        '''Ensure that scales are positive.'''
        assert options['scale_atom_radius'] > 0, "Scale of atom radius must be positive."
        assert options['scale_moment_length'] > 0, "Scale of moment length must be positive."

    def _validate_coordinates(self, options: dict):
        '''Ensure UVW and XYZ coordinate systems have appropriate entries.'''
        assert options['coordinates_xyz_position'] in [1,2,3,4], "Position code for XYZ coordinates must be 1,2,3,4."
        assert options['coordinates_uvw_position'] in [1,2,3,4], "Position code for UVW coordinates must be 1,2,3,4."

        assert options['coordinates_xyz_scale'] > 0, "Scale of XYZ coordinates must be a positive value."
        assert options['coordinates_uvw_scale'] > 0, "Scale of UVW coordinates must be a positive value."


    ### Override attribute setting to allow for validation
    def __setattr__(self, name, value, validate: bool=True):
        '''Attribute setter with validation (default).
        Validation involves datatypes, values, and values relations.'''
        # If validation is not needed, just set the attribute
        if not validate:
            self.__dict__[name] = value
            return
        
        # Check if attributes exists
        if name not in [att[0] for att in self.__attribute_defaults]:
            self.logger.warning(f"Not allowed to set field '{name}'.")
            return

        # Validate data types
        field_type = type(self.__dict__[name])
        if not isinstance(value, field_type):
            raise TypeError(f"Attribute '{name}' must be of type '{field_type.__name__}', got '{type(value).__name__}' instead.")
        
        # Validate data values of the new configuration
        options_new = self.__dict__.copy()
        options_new[name] = value

        if name.startswith('boundaries_v') and 'lim' in name:
            self._validate_boundaries_limits(options_new)
        elif name.startswith('boundaries_v') and (name[-1] in 'xyz'):
            self._validate_boundaries_span(options_new)
        elif name.startswith('scale_'):
            self._validate_scales(options_new)
        elif name.startswith('coordinates'):
            self._validate_coordinates(options_new)
    
        
        # If everything is ok, set the attribute
        self.__dict__[name] = value

        # And emit appropriate signal
        if name.startswith('light_dir'):
            self.events.light_dir(type='light_dir', value=value)
        elif name == 'light_headlight_on':
            self.events.headlight(type='headlight', value=value)
        elif name.startswith('boundaries_v'):
            self.events.boundaries()
        elif name.startswith('scale_'):
            self.events.scale()
        elif name.startswith('coordinates'):
            self.events.coordinates()

    def set(self, **kwargs):
        for name, value in kwargs.items():
            self.__setattr__(name, value)

        return

    @property
    def logger(self) -> logging.Logger:
        '''Logger of the class'''
        return self._logger

    ### LIGHT ###
    @property
    def light_dir(self) -> tuple[float, float, float, float]:
        '''Direction of the light source in the scene.'''
        return tuple([self.__getattribute__(f'light_dir_{axis}') for axis in 'xyzw'])


    ### BOUNDARIES ###
    @property
    def boundaries_v1(self) -> tuple[float, float, float]:
        '''First normal vector of the supercell boundaries.'''
        return tuple([self.__getattribute__(f'boundaries_v1_{axis}') for axis in 'xyz'])
    
    @property
    def boundaries_v2(self) -> tuple[float, float, float]:
        '''Second normal vector of the supercell boundaries.'''
        return tuple([self.__getattribute__(f'boundaries_v2_{axis}') for axis in 'xyz'])
    
    @property
    def boundaries_v3(self) -> tuple[float, float, float]:
        '''Third normal vector of the supercell boundaries.'''
        return tuple([self.__getattribute__(f'boundaries_v3_{axis}') for axis in 'xyz'])
    
    @property
    def boundaries_v1_lims(self) -> tuple[float, float]:
        '''Limits of the supercell along the first vector.'''
        return [self.boundaries_v1_lim_1, self.boundaries_v1_lim_2]
    @property
    def boundaries_v2_lims(self) -> tuple[float, float]:
        '''Limits of the supercell along the second vector.'''
        return [self.boundaries_v2_lim_1, self.boundaries_v2_lim_2]
    @property
    def boundaries_v3_lims(self) -> tuple[float, float]:
        '''Limits of the supercell along the third vector.'''
        return [self.boundaries_v3_lim_1, self.boundaries_v3_lim_2]
    
    def get_bbox(self) -> np.ndarray:
        '''Get the bounding box of the supercell.'''
        return np.array([self.boundaries_v1_lims, 
                         self.boundaries_v2_lims, 
                         self.boundaries_v3_lims], dtype=float)

    def set_bbox(self, bbox: np.ndarray):
        assert np.shape(bbox) == (3,2), f"bbox must have shape (3,2), given: {bbox=}"
        self.boundaries_v1_lim_1 = bbox[0][0]
        self.boundaries_v1_lim_2 = bbox[0][1]
        self.boundaries_v2_lim_1 = bbox[1][0]
        self.boundaries_v2_lim_2 = bbox[1][1]
        self.boundaries_v3_lim_1 = bbox[2][0]
        self.boundaries_v3_lim_2 = bbox[2][1]

    def __str__(self):
        ret  = '<OptionsValidator\n'
        ret += f'\t light_dir={self.light_dir}, headlight_on={self.light_headlight_on}, light_move_step={self.light_move_step}\n'
        ret += f'\t boundaries_v1={self.boundaries_v1}, boundaries_v1_lims={self.boundaries_v1_lims}\n'
        ret += f'\t boundaries_v2={self.boundaries_v2}, boundaries_v2_lims={self.boundaries_v2_lims}\n'
        ret += f'\t boundaries_v3={self.boundaries_v3}, boundaries_v3_lims={self.boundaries_v3_lims}>'
        return ret

class SelectionManager(object):
    selected_ids = []

    def __init__(self):
        self.selected_ids = []

        # Create events list
        emitters = EmitterGroup(source=self, new=Event, cleared=Event)
        self.__setattr__('events', emitters)
        
        self.events.connect(self._event_debugger)
        
    def _event_debugger(self, event):
        print('SELECTIONMANAGER: Event triggered')
        print(event.__repr__())

    def add(self, id):
        """Add element with `id` to the selection."""
        self.selected_ids.append(id)
        self.events.new(type='new_selection', value=id)

    # def remove(self, id):
    #     """Remove element with `id` from the selection."""
    #     self.selected_ids.remove(id)

    def clear(self):
        """Cllear all selected elements."""
        self.selected_ids = []
        self.events.cleared(type='clear')

class AdvancedVispySupercellPlotter(object):
    '''
    Vispy is a 3D, ray-tracing, lightweight renderer.

    Here plotting will be done differently due to heavy usage of instanced mesh.

    Atoms, moments, bonds, DM interactions will have their own unique meshes, that will get propagated by symmetry transformations.
    Extent of the propagation needs to be controlled by the bounding box.
    Unique meshes could be anything -> get the cow for atoms, giraffe for moments.
    
    '''

    _atoms: visuals.InstancedMesh = None
    _atom_descriptions: list[str] = []
    _shaders: list[ShadingFilter] = []

    def __init__(self, sws: 'SpinW', config: dict={}):
        # This is too long
        from vispy import use

        use(gl='gl+')

        self.spinw = sws
        self.crystal = sws.crystal

        self.logger = logging.getLogger('SupercellPlotter')
        # self.logger.setLevel('INFO')

        # Init vispy objects
        self.canvas = SceneCanvas(bgcolor='white', show=True)
        self.view = ViewBox()
        self.canvas.central_widget.add_widget(self.view)
        self.view.camera = TurntableCamera()

        # self.view = self.canvas.central_widget.add_view()
        # self.view.camera = TurntableCamera()
        ##############################################################################################
        #### Configuration file
        self.config = OptionsManager()

        ### Light configuration
        self._light_transform = deepcopy(self.view.camera.transform) 
        # It's easier to handle the light direction transform by inverse mapping first
        self._light_dir_inv = self._light_transform.imap(self.config.light_dir)

        self.config.events.light_dir.connect(self.update_light_dir)
        self.config.events.headlight.connect(self.update_headlight)
        self.update_headlight() # Need to activate the headlight on the launch

        ### Boundaries configuration
        self.config.events.boundaries.connect(self.draw_supercell)
        self.config.events.scale.connect(self.draw_supercell)

        # Enable object picking
        self.selector = SelectionManager()
        self.selector.events.cleared.connect(self.draw_supercell)

        # Pressing events
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.key_press.connect(self.on_key_press)

        self.timer = Timer(interval=1./60)
        self.timer.connect(self.update_frame)


        # self._atoms = list()
        # self._atom_descriptions = list()
        self._shaders = list()
        self._prepare_meshes()

        self.draw_supercell()
        self.draw_coordinate_system()

    def plot(self, plot_options: dict[str, Any]):
        self.config.set(**plot_options)
        return self.draw_supercell()
    
    def deploy(self):
        return self.launch()

    def launch(self, plot_options: dict={}):
        # Update the scene in this functions, so it triggers all connected events
        self.view.camera.center = self._structure_center
        self.view.camera.distance = 2*self._largest_distance / np.tan(np.radians(self.view.camera.fov)/2)
        self.logger.warning(f'Center = {self._structure_center}')
        self.logger.warning(f'Distance = {self._largest_distance}')
        
        self.canvas.app.run()
    
    def default_shading_filter(self):
        sf = ShadingFilter(shading='smooth', shininess=20, ambient_coefficient=(0,0,0,0))
        sf.light_dir = self.config.light_dir[:3]
        self._shaders.append(sf)

        return sf
        
    def update_frame(self, event):
        time = self.timer.elapsed
        print(f'{time=}')

        atoms, edges = self.get_objects_in_supercell()

        np.random.seed(42)  # For reproducibility
        eigenvector = 0.2*( np.random.rand(len(atoms), 3) - 0.5 )  # Random polarization vectors
        frequency = 3.8  # Frequency [Hz]


        pos_new = self.crystal.uvw2xyz([atom.r for atom in atoms]) + eigenvector*np.sin(2*np.pi*time*frequency)
        self._atoms_visual.instance_positions = pos_new
        self._moments_visual.instance_positions = pos_new

        # pos = self.crystal.uvw2xyz([atom.r for atom in atoms])
        # sizes = np.array([atom.radius for atom in atoms]) * self.config.scale_atom_radius
        # transforms = [np.diag([ss,ss,ss]) for ss in sizes]
        # colors = np.array([atom.color for atom in atoms])

        # rotations = []
        # for atom in atoms:
        #     rotation = np.power(atom.s, 0.3) * np.linalg.inv(self.spinw.rot_Rprime(atom.m))
        #     rotations.append(rotation)


        # try:
        #     # Need to block updating until all positions/transforms/colors are set
        #     with self._atoms_visual.events.data_updated.blocker():



        # for obj in self._objects:
        #     obj.transform = STTransform(translate=np.array([0.2,0,0])*np.sin(2*np.pi*time/10))*obj.transform
        
    
    ### LIGHT HANDLING
    def update_light_dir(self, event=None):
        if self.config.light_headlight_on:
            self._light_dir_inv = self._light_transform.imap(self.config.light_dir)
            self._update_shaders_with_headlight()
        else:
            for sf in self._shaders:
                sf.light_dir = self.config.light_dir[:3]

        self.canvas.scene.update()

    def update_headlight(self, event=None):
        if self.config.light_headlight_on:
            self.view.scene.transform.changed.connect(self._update_shaders_with_headlight)
            self._update_shaders_with_headlight()
        else:
            self.view.scene.transform.changed.disconnect(self._update_shaders_with_headlight)
            for sf in self._shaders:
                sf.light_dir = self.config.light_dir[:3]

        self.canvas.scene.update()

    def _update_shaders_with_headlight(self, event=None):
        transform = self.view.camera.transform
        for sf in self._shaders:
            sf.light_dir = transform.map(self._light_dir_inv)[:3]

    ### MAIN REDRAWING FUNCTION
    def draw_supercell(self, event=None):
        '''Draw the supercell with all atoms, edges and moments.'''
        # TODO  magnetic moment length scale
        atoms, edges = self.get_objects_in_supercell()
          
        ### Atoms
        pos = self.crystal.uvw2xyz([atom.r for atom in atoms])
        sizes = np.array([atom.radius for atom in atoms]) * self.config.scale_atom_radius
        transforms = [np.diag([ss,ss,ss]) for ss in sizes]
        colors = np.array([atom.color for atom in atoms])

        rotations = []
        for atom in atoms:
            rotation = np.power(atom.s, 0.3) * np.linalg.inv(self.spinw.rot_Rprime(atom.m)) # Scale arro lengths arbitrarily
            rotations.append(rotation)

        ### Cell edges
        edges = self.crystal.uvw2xyz(edges)
        edge_dir = edges[:,1,:] - edges[:,0,:]
        edge_pos = 0.5*(edges[:,1,:] + edges[:,0,:])
        edge_rot = RfromZ(edge_dir)
        edge_transforms = [np.diag([0.1,0.1,ll]) for ll in np.linalg.norm(edge_dir, axis=-1)]
        edge_colors = np.repeat([colors[0]], len(edge_pos), axis=0)

        try:
            # Need to block updating until all positions/transforms/colors are set
            with self._atoms_visual.events.data_updated.blocker():
                self._atoms_visual.instance_positions = pos
                self._atoms_visual.instance_transforms = transforms
                self._atoms_visual.instance_colors = np.array(colors)/256.0

                self._atoms_visual.unfreeze()
                self._atoms_visual.instance_descriptions = atoms
                self._atoms_visual.freeze()

            self._atoms_visual.events.data_updated()

            # Magnetic moments
            with self._moments_visual.events.data_updated.blocker():
                self._moments_visual.instance_positions = pos
                self._moments_visual.instance_transforms = rotations
                self._moments_visual.instance_colors = np.array(colors)/256.0

            self._moments_visual.events.data_updated()

            # Celle edges
            with self._edge_visual.events.data_updated.blocker():
                self._edge_visual.instance_positions = edge_pos
                self._edge_visual.instance_transforms = edge_transforms
                self._edge_visual.instance_colors = np.array(edge_colors)/256.0

                self._edge_visual.unfreeze()
                self._edge_visual.instance_descriptions = atoms
                self._edge_visual.freeze()

            self._edge_visual.events.data_updated()
        

            self._structure_center = np.average(pos, axis=0)
            self._largest_distance = np.abs(pos-self._structure_center).max()

        except Exception as e:
            self.logger.error(traceback.format_exc())


        ### Edges


        return self._atoms_visual, self._moments_visual

    def draw_coordinate_system(self) -> dict[str, visuals.Compound]:
        '''Draw and handle coordinate systems.'''

        axes_XYZ = self.plot_arrows(positions=np.zeros((3,3)), directions=np.eye(3), colors=np.zeros((3,3)))    # black XYZ
        axes_UVW = self.plot_arrows(positions=np.zeros((3,3)), directions=self.crystal.A.T, colors=np.eye(3))   # rgb ABC
        axes = {'xyz': axes_XYZ, 'uvw': axes_UVW}


        def get_translation_scale(axis_name: str):
            '''axis_name: xyz, uvw'''
            pos_code = self.config.__getattribute__('coordinates_'+axis_name+'_position')
            offset_x = self.config.__getattribute__('coordinates_'+axis_name+'_offset_x')
            offset_y = self.config.__getattribute__('coordinates_'+axis_name+'_offset_y')
            scale = self.config.__getattribute__(f'coordinates_{axis_name}_scale')

            ww, wh = self.canvas.size
            codes_mapping = {
                1: [0, 0],
                2: [ww, 0],
                3: [ww, wh],
                4: [0, wh],
            }
            position = np.array(codes_mapping[pos_code], dtype=int)
            position += [offset_x, offset_y]

            return position, scale

        # add shaders of axes to self._shaders to update light direction
        for ax_name, ax in axes.items():
            ax.parent = self.view

            tr, ss = get_translation_scale(ax_name)
            ax.transform = STTransform(translate=tr, scale=[ss]*3).as_matrix()

            # add shader to list?


        def position_reference_system(event = None):
            cam = self.view.camera
            for ax_name, ax in axes.items():
                ax.transform.reset()

                ax.transform.rotate(cam.roll, (0, 0, 1))
                ax.transform.rotate(cam.elevation, (1, 0, 0))
                ax.transform.rotate(cam.azimuth, (0, 1, 0))

                tr, ss = get_translation_scale(ax_name)
                ax.transform.scale((ss, ss, 1e-5))    # Why does this translation is 3-tuple now? Why `z`has to ba small but non-zero?
                ax.transform.translate(tr)

                ax.update()

        self.view.scene.transform.changed.connect(position_reference_system)
        self.canvas.events.resize.connect(position_reference_system)

        return axes
    
    def on_key_press(self, event):
        print(event.__repr__())
        ### PLAY/PAUSE 
        if event.key == 'p':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

        ### ATOM RADIUS SCALE
        if event.key == 'a':
            if 'Shift' in event.modifiers:
                self.config.scale_atom_radius -= 0.2
            else:
                self.config.scale_atom_radius += 0.2
        ### LIGHT MOVING
        if event.key == 'q':
            if 'Shift' in event.modifiers:
                self.config.light_dir_x += self.config.light_move_step
            else:
                self.config.light_dir_x -= self.config.light_move_step
        if event.key == 'w':
            if 'Shift' in event.modifiers:
                self.config.light_dir_y += self.config.light_move_step
            else:
                self.config.light_dir_y -= self.config.light_move_step
        if event.key == 'e':
            if 'Shift' in event.modifiers:
                self.config.light_dir_z += self.config.light_move_step
            else:
                self.config.light_dir_z -= self.config.light_move_step

        if event.key == 'l':
            self.config.light_headlight_on = not self.config.light_headlight_on


        ### PERSEPECTIVE
        if event.key == 'r':
            fov_move_step = 2
            fov_before = self.view.camera.fov
            if 'Shift' in event.modifiers:
                self.view.camera.fov += fov_move_step
            else:
                self.view.camera.fov -= fov_move_step

            # Update distance to keep the same size of the structure
            self.view.camera.distance *= np.tan(np.radians(fov_before)/2) / np.tan(np.radians(self.view.camera.fov)/2)

        ### BOUNDARIES
        if (event.key == 'x'):
            if ('Alt' not in event.modifiers) and ('Shift' not in event.modifiers):
                self.config.boundaries_v1_lim_2 += 0.5
            if ('Alt' not in event.modifiers) and ('Shift' in event.modifiers):
                self.config.boundaries_v1_lim_2 -= 0.5
            if ('Alt' in event.modifiers) and ('Shift' in event.modifiers):
                self.config.boundaries_v1_lim_1 += 0.5
            if ('Alt' in event.modifiers) and ('Shift' not in event.modifiers):
                self.config.boundaries_v1_lim_1 -= 0.5
        if (event.key == 'y'):
            if ('Alt' not in event.modifiers) and ('Shift' not in event.modifiers):
                self.config.boundaries_v2_lim_2 += 0.5
            if ('Alt' not in event.modifiers) and ('Shift' in event.modifiers):
                self.config.boundaries_v2_lim_2 -= 0.5
            if ('Alt' in event.modifiers) and ('Shift' in event.modifiers):
                self.config.boundaries_v2_lim_1 += 0.5
            if ('Alt' in event.modifiers) and ('Shift' not in event.modifiers):
                self.config.boundaries_v2_lim_1 -= 0.5
        if (event.key == 'z'):
            if ('Alt' not in event.modifiers) and ('Shift' not in event.modifiers):
                self.config.boundaries_v3_lim_2 += 0.5
            if ('Alt' not in event.modifiers) and ('Shift' in event.modifiers):
                self.config.boundaries_v3_lim_2 -= 0.5
            if ('Alt' in event.modifiers) and ('Shift' in event.modifiers):
                self.config.boundaries_v3_lim_1 += 0.5
            if ('Alt' in event.modifiers) and ('Shift' not in event.modifiers):
                self.config.boundaries_v3_lim_1 -= 0.5

    def on_mouse_press(self, event):
        '''Handle object selection.
        
        Only atoms and bonds can be selected.
        
        Notes
        -----
        If more complex object selection ways are needed I could implement a selection manager. 
        It would hold ids of objects that are selected and handle color changes etc.
        '''
        print('CLICK !!!')
        # All interactive objects are InstancedMeshes
        clicked_object = self.canvas.visual_at(event.pos)

        # Check if object has decsirption
        if isinstance(clicked_object, visuals.InstancedMesh) and ('instance_descriptions' in clicked_object.__dict__):
            id = self._get_mesh_id_from_eventpos(event.pos, clicked_object)

            instance_pos = clicked_object.instance_positions[id]
            colors = clicked_object.instance_colors
            colors[id] = [1, 0, 0, 1]  # Highlight the clicked instance in red
            clicked_object.instance_colors = colors

            print("DESCRIPTION", clicked_object.instance_descriptions[id])
        else:
            self.selector.clear()
       
    def _get_mesh_id_from_eventpos(self, pos, mesh):
        event_pos = np.array([pos[0], pos[1], 0, 1])  

        # Translate each position to corresponding 2d canvas coordinates
        min = 10000
        id = None
        for n,instance in enumerate(mesh.instance_positions):
            instance_pos = mesh.get_transform(map_from="visual", map_to="canvas").map(instance)
            instance_pos /= instance_pos[3:]

            # Find the closest position to the clicked position
            # Not minding z axis -> maybe we can use some sensible weighing scheme with z?
            temp_min = np.linalg.norm(
                np.array(event_pos[:2]) - np.array(instance_pos[:2])
            )
            if temp_min < min:
                min = temp_min
                id = n

        return id

    ##########################################################

    def get_objects_in_supercell(self) -> tuple[list['Atom'], np.ndarray]:
        '''Find all objects that fit within the boundaries
        
        Returns
        -------
        atoms: list[Atom]
        edges: np.ndarray (N,2,3)
            Edges (N) with start and end point (2) coordinates (3).
        '''
        EPS = 1e-8
        bbox = self.config.get_bbox()

        # TODO take boundaries vectors into account
        # -> Take corners of the polyhedron defined by the boundaries vectors
        #    and treat them as limits, then make a np.ndarray covering that
        #    region and find where the unit cell fits.
        # -> Change paradigm, as bbox is really defined by abc,
        #    and add funcionality of clipping planes.
        # Atoms have to be taken in negative cells as well
        ext_atoms = np.floor(bbox).astype(int)
        # Edges only for full cells
        ext_edges = np.trunc(bbox).astype(int)

        ### EDGES
        edges = []
        it_nx = range(ext_edges[0][0], ext_edges[0][1]+1)
        it_ny = range(ext_edges[1][0], ext_edges[1][1]+1)
        it_nz = range(ext_edges[2][0], ext_edges[2][1]+1)

        for ny in it_ny:
            for nz in it_nz:
                edges.append([[ext_edges[0][0],ny,nz], [ext_edges[0][1],ny,nz]])
        for nx in it_nx:
            for nz in it_nz:
                edges.append([[nx,ext_edges[1][0],nz], [nx,ext_edges[1][1],nz]])
        for nx in it_nx:
            for ny in it_ny:
                edges.append([[nx,ny,ext_edges[2][0]], [nx,ny,ext_edges[2][1]]])

        edges = np.array(edges, dtype=float)

        ### ATOMS
        atoms = []
        it_nx = range(ext_atoms[0][0], ext_atoms[0][1]+1)
        it_ny = range(ext_atoms[1][0], ext_atoms[1][1]+1)
        it_nz = range(ext_atoms[2][0], ext_atoms[2][1]+1)
        low_bound = bbox[:,0] - EPS
        high_bound = bbox[:,1] + EPS
        for zcen in it_nz:
            for ycen in it_ny:
                for xcen in it_nx:
                    for atom in self.crystal.atoms_all:
                        atom_candidate = deepcopy(atom)
                        n_uvw = np.array([xcen, ycen, zcen])

                        atom_candidate.r += n_uvw

                        if atom.is_mag:
                            rotated_spin = np.dot(self.spinw.rot_Rn(n_uvw), atom.m.T)
                            atom_candidate.m = rotated_spin
                        
                        if all(atom_candidate.r > low_bound) and all(atom_candidate.r < high_bound):
                            atoms.append(atom_candidate)

                    # atoms_pos_supercell[icell*natoms:(icell+1)*natoms,:] = atoms_pos_unit_cell + np.array([xcen, ycen, zcen])


                    # atoms_mom_supercell[icell*natoms:(icell+1)*natoms,:] = rotated_spins.T

                    # icell += 1

        return atoms, edges
    
    def _prepare_meshes(self, names: list[str]=[]):        
        '''Prepare base visuals that will be used to depict the crystal.
        
        '''
        # So now the main deal is that we will have just one mesh,
        # and atoms positions and their size will be in the multiple transforms

        ### ATOMS ###


        # Base mesh is dinosaur
        # from vispy.io import imread, load_data_file, read_mesh

        # mesh_path = load_data_file('spot/spot.obj.gz')
        # texture_path = load_data_file('spot/spot.png')
        # vertices, faces, normals, texcoords = read_mesh(mesh_path)
        # vertices *= 2  # Scale the mesh to make it larger
        # ### The texture needs to be attached to the Instanced mesh
        # texture = np.flipud(imread(texture_path))
        # texture_filter = TextureFilter(texture, texcoords)

        # Base mesh is sphere
        N = 8
        sphere = visuals.Sphere(radius=1, method='latitude', cols=N, rows=N)
        mesh = sphere._mesh.mesh_data
        vertices, faces = mesh.get_vertices(), mesh.get_faces()

        self._atoms_visual = visuals.InstancedMesh(vertices=vertices, faces=faces,
                                        instance_positions = [0,0,0],
                                        instance_transforms = np.eye(3),
                                        instance_colors= [1,1,1],
                                        parent=self.view.scene)
        
        self._atoms_visual.interactive = True
        # self._atoms_visual.attach(Alpha(0.75))
        self._atoms_visual.attach( self.default_shading_filter() )
        # self._atoms_visual.attach(texture_filter)

        ### MAGNETIC MOMENTS ###
        self.spin_scale = 1
        self.arrow_width = 0.1
        self.arrow_head_size = 3

        tail_width = 0.1
        head_width = 3*tail_width

        cap_res = 1
        ang_res = 32
        # Length of the arrows is such that they stick out of the atoms
        # length = max([1.5*atom.radius/atom.s for atom in self.crystal.atoms_unique if atom.is_mag])
        points, radii = self._arrow_points(length=1.5, 
                                            tail_width=tail_width, 
                                            head_length=0.5, head_width=head_width, 
                                            overhang=-0.05, 
                                            cap_res=cap_res, head_res=16)
        arrow = visuals.Tube(points=points, radius=radii, tube_points=ang_res)
        mesh = arrow.mesh_data
        self._moments_visual = visuals.InstancedMesh(vertices=mesh.get_vertices(), faces=mesh.get_faces(),
                                                     instance_positions = [0,0,0],
                                                     instance_transforms = np.eye(3),
                                                     instance_colors= [1,1,1],
                                                     parent=self.view.scene)
                
        self._moments_visual.interactive = True
        self._moments_visual.attach(self.default_shading_filter())

        ### CELL EDGES ###
        edge = visuals.Tube(points=[[0,0,-1e-3], [0,0,0], [0,0,1], [0,0,1+1e-3]], radius=[0,1,1,0], tube_points=ang_res)
        mesh = edge.mesh_data
        self._edge_visual = visuals.InstancedMesh(vertices=mesh.get_vertices(), faces=mesh.get_faces(),
                                                     instance_positions = [0,0,0],
                                                     instance_transforms = np.eye(3),
                                                     instance_colors= [1,1,1],
                                                     parent=self.view.scene)
                
        self._edge_visual.attach(self.default_shading_filter())

        ### BONDS ###
        
        return
        
    def plot_labels(self,
                    positions: np.ndarray,
                    labels: np.ndarray,
                    colors: np.ndarray):
        
        colors = colors/255.
        # Labels
        # font size was nice and tricky in original vispy implementation
        obj = visuals.Text(pos=positions, 
                                 parent=self.canvas.scene, 
                                 text=labels, 
                                 color="white", 
                                 font_size=12)
        obj.parent = self.view.scene

        self._objects.append(obj)
        return obj

    @staticmethod
    def _arrow_points(length, tail_width, head_length, head_width, 
                      overhang=0.05, head_Bezier_control=[[0.4], [0.8]],
                      cap_res=1, head_res=16) -> tuple[np.ndarray, np.ndarray]:
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
        points_head = [[0,0,x] for x in length-head_length + Bx*head_length] # Map Bx (0,1) to (length-head_length, length)
        radii_head = list(By*head_width)


        return (points_cap+points_tail+points_head, radii_cap+radii_tail+radii_head)
    
    def plot_arrows(self, positions: np.ndarray, directions: np.ndarray, colors: np.ndarray) -> visuals.Compound:
        '''Plot arrows at `positions` with `directions` and `colors`.
        Arrows point from `position` to `position + direction`.'''
        # colors = colors/255.

        # Magnetic moments from magnetic atoms only
        # def plot_magnetic_structure(self, canvas_scene, mj, pos, colors):
        self.spin_scale = 1
        self.arrow_width = 0.1
        self.arrow_head_size = 3

        cap_res = 16
        tail_width = 0.1
        head_width = 3*tail_width

        cap_res = 1
        ang_res = 64

        # obj = visuals.XYZAxis(parent=self.canvas.scene)

        objects = []
        for position, direction, color in zip(positions, directions, colors):
            r, th, phi = cartesian2spherical(direction)
            points, radii = self._arrow_points(length=r, 
                                               tail_width=tail_width, 
                                               head_length=0.5, head_width=head_width, 
                                               overhang=-0.05, 
                                               cap_res=cap_res, head_res=16)
            a = visuals.Tube(points=points, radius=radii, tube_points=ang_res)
            mesh = a.mesh_data
            m1, m2 = MatrixTransform(), MatrixTransform()
            m1.rotate(np.degrees(th), (0,1,0))
            m2.rotate(np.degrees(phi), (0,0,1))
            mesh_rot = (m2.matrix @ m1.matrix)[:3,:3] @ mesh.get_vertices().T
            mesh.set_vertices(mesh_rot.T)
            obj = visuals.Mesh(vertices=mesh.get_vertices(), faces=mesh.get_faces(), color=color)
            obj.attach(self.default_shading_filter())
            
            # m2.translate(position)
            # obj.transform = m2 * m1
            # obj.parent = self.view.scene

            objects.append(obj)

        return visuals.Compound(objects)
    
    def plot_lines(self,
                   lines: np.ndarray,
                   colors: np.ndarray,
                   width: float=0.05,
                   alpha: float=1):
        '''
        width in px
        '''
        colors = colors/255.
        ang_res = 32
        for line, color in zip(lines, colors):
            ### With tube
            # width = width/10
            # dir = np.array(line[1])-np.array(line[0])
            # dir /= np.linalg.norm(dir)
            # points = [line[0], line[0]+dir*width/2, line[1]-dir*width/2, line[1]]
            # l = visuals.Tube(points=points, radius=[1e-10,width,width,1e-10], tube_points=ang_res, shading='flat')

            ### With line
            obj = visuals.Line(line, width=width, color=color, antialias=True)
            obj.parent = self.view.scene

            self._objects.append(obj)

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
            obj.parent = self.view.scene

            self._objects.append(obj)

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
        self.logger.info(f'Plotting ellipsoid with matrix \n {M}')
        self.logger.info(f'Ellipsoid evals = {e_vals}')
        self.logger.info(f'Ellipsoid evecs = {e_vecs}')

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
            obj.parent = self.view.scene

            self._objects.append(obj)
            
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

        def arrow_points(length, tail_width, head_length, head_width, overhang, cap_res) -> tuple[np.ndarray, np.ndarray]:
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

        # ###
        # points, radii = arrow_points(length=2, tail_width=tail_width, head_length=0.5, head_width=head_width, overhang=0, cap_res=3*cap_res)
        # a2 = visuals.Tube(points=points, radius=radii, tube_points=ang_res, shading='smooth')
        
        # a2.transform = STTransform(translate=[2,0,0])
        # self.view.add(a2)

        # ###
        # points, radii = arrow_points(length=2, tail_width=tail_width, head_length=0.5, head_width=head_width, overhang=0, cap_res=6*cap_res)
        # a2 = visuals.Tube(points=points, radius=radii, tube_points=ang_res)
        # a2l = visuals.Text(text='shading="smooth"', font_size=15)
        
        # a2.transform = STTransform(translate=[4,0,0])
        # a2l.transform = a2.transform
        # self.view.add(a2)
        # self.view.add(a2l)

        # ###
        # points, radii = arrow_points(length=3, tail_width=tail_width, head_length=0.5, head_width=head_width, overhang=0, cap_res=6*cap_res)
        # a4 = visuals.Tube(points=points, radius=radii, tube_points=ang_res)
        # mesh = a4.mesh_data

        # a4 = visuals.Mesh(vertices=mesh.get_vertices(),
        #                     faces=mesh.get_faces(), color='green')
        
        # shading_filter = ShadingFilter(shading='smooth', shininess=0)
        # a4.attach(shading_filter)

        # a4.transform = STTransform(translate=[6,0,0])

        # self.view.add(a4)


### TESSTING

# canvas = SceneCanvas(bgcolor='white', show=True)
# view = ViewBox()
# canvas.central_widget.add_widget(view)
# view.camera = TurntableCamera()
# sf = None

# sf = ShadingFilter(shading='smooth', shininess=10, ambient_coefficient=(0,0,0,0))


# def plot_balls(positions: np.ndarray, 
#                sizes: np.ndarray, 
#                colors: np.ndarray):
#     objects = []
#     for position, size, color in zip(positions, sizes, colors):
#         sphere = visuals.Sphere(radius=size, method='latitude', color=color)
#         mesh = sphere._mesh.mesh_data
#         obj = visuals.Mesh(vertices=mesh.get_vertices(),faces=mesh.get_faces(), color=color)
        
#         # obj.attach(sf)

#         obj.transform = STTransform(translate=position)

#         objects.append(obj)

#     return objects
    

# def spring_points(r1, r2, n_turns=10, spring_thickness=0.5, turn_radius = 1,
#                   spring_resolution: int=32) -> tuple[np.ndarray, np.ndarray]:
#     '''
#     Returns points and radii of a spring.
#     '''

#     spring_length = np.linalg.norm(r2 - r1)
#     t = np.linspace(0, n_turns, n_turns*spring_resolution)

#     if isinstance(turn_radius, list):
#         turn_radius = np.interp(t, n_turns*np.linspace(0,1,len(turn_radius)), turn_radius)
#     if isinstance(spring_thickness, list):
#         spring_thickness = np.interp(t, n_turns*np.linspace(0,1,len(spring_thickness)), spring_thickness)

#     x = turn_radius * np.cos(2*np.pi*t)
#     y = turn_radius * np.sin(2*np.pi*t)
#     z = t*spring_length/n_turns  # Linear increase in height

#     points = np.array([x, y, z]).T
#     radii = spring_thickness*np.ones(len(t))

#     return points, radii

# r1 = np.array([0, 0, 0])
# r2 = np.array([0, 0, 15])

# obj.parent = None
# balls[0].parent = None
# balls[1].parent = None

# points, radii = spring_points(r1, r2, n_turns=15, turn_radius=[0,0.5,1,1,0], spring_thickness=0.2)
# a = visuals.Tube(points=points, radius=radii, tube_points=8)
# mesh = a.mesh_data
# obj = visuals.Mesh(vertices=mesh.get_vertices(), faces=mesh.get_faces(), color='blue')
# obj.attach(sf)
# view.add(obj)

# balls = plot_balls(positions=[r1,r2], colors=['red', 'green'], sizes=[0.5, 0.5])
# view.add(balls[0])
# view.add(balls[1])

bl_info = {
    "name": "Mozi's Facemocap(receiver)",
    "author": "Mozi,DeepSeek and You",
    "version": (0, 12),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Mozi's FaceCapture",
    "description": "Converting facial expressions to controller data",
    "category": "Animation",
}

import bpy
import math
import socket
import json
import threading
import queue
import csv
import os
from mathutils import Vector
from bpy.types import Operator, Panel
from bpy.props import StringProperty, IntProperty, BoolProperty, PointerProperty
from bpy_extras.io_utils import ImportHelper

# ======================== Global Configuration ========================
category = "Mozi's FaceCapture"
UDP_IP = '127.0.0.1'
UDP_PORT = 12345
sock = None
is_receiving = False
udp_thread = None
data_queue = queue.Queue()
DEBUG_MAX_LINES = 20
BASE_ARMATURE_NAME = "FaceCapture_Rig"

controls = {
    'mouth':        'MFC_Mouth',
    'left_eyelid':  'MFC_LeftEyelid',
    'right_eyelid': 'MFC_RightEyelid',
    'left_pupil':   'MFC_LeftPupil',
    'right_pupil':  'MFC_RightPupil',
    'left_brow':    'MFC_LeftBrow',
    'right_brow':   'MFC_RightBrow',
    'head':         'MFC_Head',
    'teeth':        'MFC_Teeth'
}

control_groups = [
    ('right_brow', 'left_brow'),
    ('right_eyelid', 'left_eyelid'),
    ('right_pupil', 'left_pupil'),
    ('mouth',),
    ('teeth',),
    ('head',)
]

PUPIL_MOVE_RANGE = 0.1

# ======================== Core Functionality ========================
def get_armature(context, create_new=False):
    """Get or create armature"""
    if create_new:
        armature_data = bpy.data.armatures.new(BASE_ARMATURE_NAME)
        armature_obj = bpy.data.objects.new(BASE_ARMATURE_NAME, armature_data)
        context.collection.objects.link(armature_obj)
        
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        root_bone = armature_data.edit_bones.new("Root")
        root_bone.head = Vector((0, 0, 0))
        root_bone.tail = Vector((0, 0, 0.2))
        
        for index, (control_type, bone_name) in enumerate(controls.items()):
            new_bone = armature_data.edit_bones.new(bone_name)
            x_offset = index * 0.3
            new_bone.head = root_bone.head + Vector((x_offset, 0, 0.1))
            new_bone.tail = new_bone.head + Vector((0, 0.2, 0))
            new_bone.parent = root_bone
        
        bpy.ops.object.mode_set(mode='OBJECT')
        return armature_obj
    
    return context.scene.fpc_active_armature

def get_pose_bone(armature, bone_name):
    """Get pose bone"""
    if armature and bone_name in armature.pose.bones:
        return armature.pose.bones[bone_name]
    return None

def apply_facial_data(sc, armature, info, frame, auto_key=True):
    """Apply facial data to bones"""
    # 1. Mouth control
    if sc.fpc_enable_mouth:
        mouth = get_pose_bone(armature, controls['mouth'])
        if mouth:
            mouth.scale[0] = 1.0 + info.get('mouth_width', 0.0)
            mouth.scale[2] = info.get('mouth_open', 0.0)
            if auto_key:
                mouth.keyframe_insert(data_path='scale', frame=frame)

    # 2. Eyelid control
    for side in ('left', 'right'):
        if getattr(sc, f'fpc_enable_{side}_eyelid'):
            eyelid = get_pose_bone(armature, controls[f'{side}_eyelid'])
            if eyelid:
                eyelid.scale[2] = info.get(f'{side}_eyelid', 0.0)
                if auto_key:
                    eyelid.keyframe_insert(data_path='scale', frame=frame)

    # 3. Pupil control
    for side in ('left', 'right'):
        if getattr(sc, f'fpc_enable_{side}_pupil'):
            pupil = get_pose_bone(armature, controls[f'{side}_pupil'])
            if pupil:
                x = max(min(info.get(f'{side}_pupil_x', 0.0), PUPIL_MOVE_RANGE), -PUPIL_MOVE_RANGE)
                y = max(min(info.get(f'{side}_pupil_y', 0.0), PUPIL_MOVE_RANGE), -PUPIL_MOVE_RANGE)
                pupil.location = (x, y, 0)
                if auto_key:
                    pupil.keyframe_insert(data_path='location', frame=frame)

    # 4. Eyebrow control
    for side in ('left', 'right'):
        if getattr(sc, f'fpc_enable_{side}_brow'):
            brow = get_pose_bone(armature, controls[f'{side}_brow'])
            if brow:
                brow.scale[2] = 1.0 + info.get(f'{side}_brow', 0.0)
                if auto_key:
                    brow.keyframe_insert(data_path='scale', frame=frame)

    # 5. Head Control
    if sc.fpc_enable_head:
        head = get_pose_bone(armature, controls['head'])
        if head:
            head.rotation_mode = 'XYZ'
            pitch = math.radians(info.get('head_pitch', 0.0))
            yaw   = math.radians(info.get('head_yaw',   0.0))
            roll  = math.radians(info.get('head_roll',  0.0))
            head.rotation_euler = (pitch, yaw, roll)
            if auto_key:
                head.keyframe_insert(data_path='rotation_euler', frame=frame)

    # 6. Teeth Control
    if sc.fpc_enable_teeth:
        teeth = get_pose_bone(armature, controls['teeth'])
        if teeth:
            teeth.scale[2] = info.get('teeth_open', 0.0)
            if auto_key:
                teeth.keyframe_insert(data_path='scale', frame=frame)

def process_data():
    """Process queued data"""
    sc = bpy.context.scene
    frame = sc.frame_current
    armature = sc.fpc_active_armature

    while not data_queue.empty() and armature:
        try:
            info = data_queue.get_nowait()
            apply_facial_data(sc, armature, info, frame, auto_key=sc.tool_settings.use_keyframe_insert_auto)
        except Exception as e:
            print(f"Error processing data: {str(e)}")
    
    return 0.02

def udp_listener():
    """UDP listener thread"""
    global sock, is_receiving
    while is_receiving:
        try:
            data, _ = sock.recvfrom(4096)
            txt = data.decode('utf-8')
            info = json.loads(txt)
            data_queue.put(info)
            
            if bpy.context.scene.fpc_debug_show:
                debug_str = json.dumps(info, indent=2)
                lines = debug_str.split('\n')[:DEBUG_MAX_LINES]
                bpy.context.scene.fpc_debug_data = '\n'.join(lines)
                
        except Exception as e:
            print(f"Receiving Error: {str(e)}")

def start_receiving(ip, port):
    """Start UDP receiving"""
    global sock, is_receiving, udp_thread
    stop_receiving()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    is_receiving = True
    udp_thread = threading.Thread(target=udp_listener, daemon=True)
    udp_thread.start()

def stop_receiving():
    """Stop UDP receiving"""
    global sock, is_receiving
    is_receiving = False
    if sock:
        try: sock.close()
        except: pass
        sock = None

# ======================== Recording Import ========================
def parse_recording_data(filepath):
    """Parse recorded CSV data"""
    data = []
    
    # Ensure absolute path
    abs_path = bpy.path.abspath(filepath)
    if not os.path.exists(abs_path):
        print(f"File not found: {abs_path}")
        return []
    
    try:
        with open(abs_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert data types
                frame_data = {
                    'head_pitch': float(row.get('head_pitch', 0)),
                    'head_yaw': float(row.get('head_yaw', 0)),
                    'head_roll': float(row.get('head_roll', 0)),
                    'mouth_open': float(row.get('mouth_open', 0)),
                    'mouth_width': float(row.get('mouth_width', 0)),
                    'left_eyelid': float(row.get('left_eyelid', 0)),
                    'right_eyelid': float(row.get('right_eyelid', 0)),
                    'left_pupil_x': float(row.get('left_pupil_x', 0)),
                    'left_pupil_y': float(row.get('left_pupil_y', 0)),
                    'right_pupil_x': float(row.get('right_pupil_x', 0)),
                    'right_pupil_y': float(row.get('right_pupil_y', 0))
                }
                data.append(frame_data)
        return data
    except Exception as e:
        print(f"Error reading recording: {str(e)}")
        return []

class FPC_OT_ImportRecording(Operator, ImportHelper):
    """Import recording file"""
    bl_idname = "fpc.import_recording"
    bl_label = "Import Recording"
    bl_description = "Import CSV facial capture recording"
    
    filename_ext = ".csv"
    filter_glob: StringProperty(default="*.csv", options={'HIDDEN'})
    
    def execute(self, context):
        sc = context.scene
        # Store absolute path
        sc.fpc_record_file = bpy.path.abspath(self.filepath)
        return {'FINISHED'}

class FPC_OT_PlayRecording(Operator):
    """Play/Pause recording"""
    bl_idname = "fpc.play_recording"
    bl_label = "Play/Pause Recording"
    
    _timer = None
    _recording_data = []
    _current_index = 0
    _is_playing = False
    _start_frame = 0
    
    def modal(self, context, event):
        sc = context.scene
        
        # ESC to cancel
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}
            
        # Space to toggle play/pause
        if event.type == 'SPACE' and event.value == 'PRESS':
            self._is_playing = not self._is_playing
            if not self._is_playing:
                return {'PASS_THROUGH'}
        
        if event.type == 'TIMER' and self._is_playing:
            if not self._recording_data or self._current_index >= len(self._recording_data):
                self.cancel(context)
                return {'CANCELLED'}
            
            # Set current frame
            frame = self._start_frame + self._current_index
            sc.frame_set(frame)
            
            # Apply data
            armature = sc.fpc_active_armature
            if armature:
                apply_facial_data(sc, armature, self._recording_data[self._current_index], frame, auto_key=False)
            
            self._current_index += 1
            
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        sc = context.scene
        
        # Toggle play/pause
        if self._is_playing:
            self._is_playing = False
            return {'FINISHED'}
        
        if not sc.fpc_record_file:
            self.report({'ERROR'}, "Please select a recording file first")
            return {'CANCELLED'}
            
        # Load recording data
        self._recording_data = parse_recording_data(sc.fpc_record_file)
        if not self._recording_data:
            self.report({'ERROR'}, "Failed to read recording file or file is empty")
            return {'CANCELLED'}
        
        # Set start frame to current frame
        self._start_frame = sc.frame_current
        self._current_index = 0
        self._is_playing = True
        
        # Calculate timer interval based on frame rate
        fps = sc.render.fps / sc.render.fps_base
        interval = 1.0 / fps
        
        # Start timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(interval, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        self._recording_data = []
        self._current_index = 0
        self._is_playing = False

class FPC_OT_BakeRecording(Operator):
    """Bake recording to keyframes"""
    bl_idname = "fpc.bake_recording"
    bl_label = "Bake Recording"
    
    def execute(self, context):
        sc = context.scene
        
        if not sc.fpc_record_file:
            self.report({'ERROR'}, "Please select a recording file first")
            return {'CANCELLED'}
            
        # Load recording data
        recording_data = parse_recording_data(sc.fpc_record_file)
        if not recording_data:
            self.report({'ERROR'}, "Failed to read recording file or file is empty")
            return {'CANCELLED'}
            
        armature = sc.fpc_active_armature
        if not armature:
            self.report({'ERROR'}, "Please select or create an armature first")
            return {'CANCELLED'}
        
        # Save original auto keyframe setting
        original_auto_key = sc.tool_settings.use_keyframe_insert_auto
        sc.tool_settings.use_keyframe_insert_auto = True
        
        # Bake each frame
        start_frame = sc.fpc_record_start_frame
        for i, frame_data in enumerate(recording_data):
            frame = start_frame + i
            sc.frame_set(frame)
            apply_facial_data(sc, armature, frame_data, frame, auto_key=True)
        
        # Restore original setting
        sc.tool_settings.use_keyframe_insert_auto = original_auto_key
        
        self.report({'INFO'}, f"Successfully baked {len(recording_data)} frames")
        return {'FINISHED'}

# ======================== UI Panels ========================
class FPC_OT_CreateControls(bpy.types.Operator):
    bl_idname = "fpc.create_controls"
    bl_label = "Create New Facial Control Rig"

    def execute(self, context):
        new_armature = get_armature(context, create_new=True)
        context.scene.fpc_active_armature = new_armature
        self.report({'INFO'}, f"Rig created: {new_armature.name}")
        return {'FINISHED'}

class FPC_OT_Start(bpy.types.Operator):
    bl_idname = "fpc.start"
    bl_label = "Start Receiving"

    def execute(self, context):
        sc = context.scene
        start_receiving(sc.udp_ip, sc.udp_port)
        sc.fpc_receiving = True
        self.report({'INFO'}, "UDP receiving started")
        return {'FINISHED'}

class FPC_OT_Stop(bpy.types.Operator):
    bl_idname = "fpc.stop"
    bl_label = "Stop Receiving"

    def execute(self, context):
        stop_receiving()
        context.scene.fpc_receiving = False
        self.report({'INFO'}, "UDP receiving stopped")
        return {'FINISHED'}

class FPC_PT_Panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = category
    bl_label = 'Face Capture'
    bl_order = 0

    def draw(self, context):
        sc = context.scene
        layout = self.layout
        
        # Armature selection
        layout.prop_search(sc, "fpc_active_armature", sc, "objects", 
                          text="Active Armature", icon='ARMATURE_DATA')
        layout.operator('fpc.create_controls', icon='ADD')
        
        # UDP control
        layout.separator()
        layout.prop(sc, 'udp_ip')
        layout.prop(sc, 'udp_port')
        
        if sc.fpc_receiving:
            layout.operator('fpc.stop', icon='CANCEL')
        else:
            layout.operator('fpc.start', icon='PLAY')

class FPC_PT_ControlPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = category
    bl_label = 'Control Settings'
    bl_order = 1
    
    def draw(self, context):
        sc = context.scene
        layout = self.layout
        
        for group in control_groups:
            row = layout.row()
            for control in group:
                if control in controls:
                    prop_name = f'fpc_enable_{control}'
                    row.prop(sc, prop_name, 
                            text=control.replace('_', ' ').title(),
                            toggle=True)

class FPC_PT_RecordPanel(bpy.types.Panel):
    """Recording import panel"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = category
    bl_label = 'Recording Import'
    bl_order = 2
    
    def draw(self, context):
        sc = context.scene
        layout = self.layout
        
        # File selection
        row = layout.row()
        row.prop(sc, 'fpc_record_file', text="File")
        row.operator('fpc.import_recording', icon='FILE_FOLDER', text="")
        
        # Start frame setting
        layout.prop(sc, 'fpc_record_start_frame', text="Start Frame")
        
        # Action buttons
        row = layout.row(align=True)
        row.operator('fpc.play_recording', icon='PLAY' if not sc.fpc_recording_playing else 'PAUSE')
        row.operator('fpc.bake_recording', icon='KEYINGSET')

class FPC_PT_DebugPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = category
    bl_label = "Debug Information"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 3

    def draw_header(self, context):
        layout = self.layout
        layout.prop(context.scene, "fpc_debug_show", text="")

    def draw(self, context):
        layout = self.layout
        sc = context.scene
        
        if sc.fpc_debug_show:
            box = layout.box()
            if sc.fpc_debug_data:
                lines = sc.fpc_debug_data.split('\n')
                for line in lines:
                    row = box.row()
                    row.alignment = 'LEFT'
                    row.label(text=line)
            else:
                box.label(text="Waiting for data...", icon='INFO')

# ======================== Registration & Initialization ========================
classes = (
    FPC_OT_CreateControls,
    FPC_OT_Start,
    FPC_OT_Stop,
    FPC_OT_ImportRecording,
    FPC_OT_PlayRecording,
    FPC_OT_BakeRecording,
    FPC_PT_Panel,
    FPC_PT_ControlPanel,
    FPC_PT_RecordPanel,
    FPC_PT_DebugPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # UDP properties
    bpy.types.Scene.udp_ip = StringProperty(
        name="UDP IP", default=UDP_IP)
    bpy.types.Scene.udp_port = IntProperty(
        name="UDP Port", default=UDP_PORT, min=1, max=65535)
    bpy.types.Scene.fpc_receiving = BoolProperty(default=False)
    
    # Armature properties
    bpy.types.Scene.fpc_active_armature = PointerProperty(
        name="Active Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    
    # Debug properties
    bpy.types.Scene.fpc_debug_show = BoolProperty(
        name="Show Debug Info",
        default=False
    )
    bpy.types.Scene.fpc_debug_data = StringProperty(
        name="Debug Data",
        default=""
    )
    
    # Recording properties
    bpy.types.Scene.fpc_record_file = StringProperty(
        name="Recording File",
        subtype='FILE_PATH',
        description="Select recording file to import"
    )
    bpy.types.Scene.fpc_record_start_frame = IntProperty(
        name="Start Frame",
        default=1,
        description="Frame to start playback/baking from"
    )
    bpy.types.Scene.fpc_recording_playing = BoolProperty(
        name="Recording Playing",
        default=False,
        description="Is recording currently playing"
    )
    
    # Control properties
    control_props = {
        'mouth': True,
        'teeth': True,
        'head': True,
        'left_eyelid': True,
        'right_eyelid': True,
        'left_brow': True,
        'right_brow': True,
        'left_pupil': True,
        'right_pupil': True,
    }
    
    for prop, default in control_props.items():
        setattr(bpy.types.Scene, f'fpc_enable_{prop}', 
               BoolProperty(name=prop.replace('_', ' ').title(), 
               default=default))
    
    bpy.app.timers.register(process_data)

def unregister():
    stop_receiving()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Remove custom properties
    props_to_remove = [
        'udp_ip', 'udp_port', 'fpc_receiving', 'fpc_active_armature',
        'fpc_debug_show', 'fpc_debug_data', 'fpc_record_file', 
        'fpc_record_start_frame', 'fpc_recording_playing'
    ]
    
    # Remove control properties
    for prop in control_props.keys():
        prop_name = f'fpc_enable_{prop}'
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
    
    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

if __name__ == "__main__":
    register()
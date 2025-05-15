bl_info = {
    "name": "Mozi's Facemocap(receiver)",
    "author": "Mozi,DeepSeek and You",
    "version": (0, 11),
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
from mathutils import Vector
from bpy.types import Operator, Panel
from bpy.props import StringProperty, IntProperty, BoolProperty, PointerProperty

# Global Configuration
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

# 新增部位分组列表
control_groups = [
    ('right_brow', 'left_brow'),
    ('right_eyelid', 'left_eyelid'),
    ('right_pupil', 'left_pupil'),
    ('mouth',),
    ('teeth',),
    ('head',)
]

PUPIL_MOVE_RANGE = 0.1

def get_armature(context, create_new=False):
    """Get or create a skeleton (supports automatic naming)"""
    if create_new:
        # Create a new skeleton and enter edit mode
        armature_data = bpy.data.armatures.new(BASE_ARMATURE_NAME)
        armature_obj = bpy.data.objects.new(BASE_ARMATURE_NAME, armature_data)
        context.collection.objects.link(armature_obj)
        
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Creating the Root Bone
        root_bone = armature_data.edit_bones.new("Root")
        root_bone.head = Vector((0, 0, 0))
        root_bone.tail = Vector((0, 0, 0.2))
        
        # Creating Control Bones
        for index, (control_type, bone_name) in enumerate(controls.items()):
            new_bone = armature_data.edit_bones.new(bone_name)
            x_offset = index * 0.3
            new_bone.head = root_bone.head + Vector((x_offset, 0, 0.1))
            new_bone.tail = new_bone.head + Vector((0, 0.2, 0))
            new_bone.parent = root_bone
        
        bpy.ops.object.mode_set(mode='OBJECT')
        return armature_obj
    
    # Returns the currently selected skeleton
    return context.scene.fpc_active_armature

def get_pose_bone(armature, bone_name):
    """Get the pose skeleton"""
    if armature and bone_name in armature.pose.bones:
        return armature.pose.bones[bone_name]
    return None

def process_data():
    sc = bpy.context.scene
    frame = sc.frame_current
    auto_key = sc.tool_settings.use_keyframe_insert_auto
    armature = sc.fpc_active_armature

    while not data_queue.empty() and armature:
        try:
            info = data_queue.get_nowait()
            
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

        except Exception as e:
            print(f"Error processing data: {str(e)}")
    
    return 0.02

def udp_listener():
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
    global sock, is_receiving, udp_thread
    stop_receiving()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    is_receiving = True
    udp_thread = threading.Thread(target=udp_listener, daemon=True)
    udp_thread.start()

def stop_receiving():
    global sock, is_receiving
    is_receiving = False
    if sock:
        try: sock.close()
        except: pass
        sock = None

class FPC_OT_CreateControls(bpy.types.Operator):
    bl_idname = "fpc.create_controls"
    bl_label = "Creating a New Facial Control Skeleton"

    def execute(self, context):
        new_armature = get_armature(context, create_new=True)
        context.scene.fpc_active_armature = new_armature
        self.report({'INFO'}, f"Skeleton created {new_armature.name}")
        return {'FINISHED'}

class FPC_OT_Start(bpy.types.Operator):
    bl_idname = "fpc.start"
    bl_label = "Start receiving data"

    def execute(self, context):
        sc = context.scene
        start_receiving(sc.udp_ip, sc.udp_port)
        sc.fpc_receiving = True
        self.report({'INFO'}, "UDP receive started")
        return {'FINISHED'}

class FPC_OT_Stop(bpy.types.Operator):
    bl_idname = "fpc.stop"
    bl_label = "Stop receiving"

    def execute(self, context):
        stop_receiving()
        context.scene.fpc_receiving = False
        self.report({'INFO'}, "UDP reception stopped")
        return {'FINISHED'}

class FPC_PT_Panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mozi's FaceCapture"
    bl_label = 'Facial capture'

    def draw(self, context):
        sc = context.scene
        layout = self.layout
        
        # 骨架选择
        layout.prop_search(sc, "fpc_active_armature", sc, "objects", 
                          text="Current Skeleton", icon='ARMATURE_DATA')
        layout.operator('fpc.create_controls', icon='ADD')
        
        # UDP控制
        layout.separator()
        layout.prop(sc, 'udp_ip')
        layout.prop(sc, 'udp_port')
        
        if sc.fpc_receiving:
            layout.operator('fpc.stop', icon='CANCEL')
        else:
            layout.operator('fpc.start', icon='PLAY')

# 新增控制面板
class FPC_PT_ControlPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FaceCapture'
    bl_label = 'Control Settings'
    bl_order = 1  # 确保在主面板之下，debug面板之上
    
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

class FPC_PT_DebugPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FaceCapture'
    bl_label = "Debug Information"
    bl_options = {'DEFAULT_CLOSED'}

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

classes = (
    FPC_OT_CreateControls,
    FPC_OT_Start,
    FPC_OT_Stop,
    FPC_PT_Panel,
    FPC_PT_ControlPanel,  # 新增面板
    FPC_PT_DebugPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.udp_ip = StringProperty(
        name="UDP IP", default=UDP_IP)
    bpy.types.Scene.udp_port = IntProperty(
        name="UDP Port", default=UDP_PORT, min=1, max=65535)
    bpy.types.Scene.fpc_receiving = BoolProperty(default=False)
    bpy.types.Scene.fpc_active_armature = PointerProperty(
        name="Active Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    bpy.types.Scene.fpc_debug_show = BoolProperty(
        name="Display debug information",
        default=False
    )
    bpy.types.Scene.fpc_debug_data = StringProperty(
        name="Debug Data",
        default=""
    )
    
    # 添加启用属性
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
    
    # 正确的属性注册方式
    for prop, default in control_props.items():
        setattr(bpy.types.Scene, f'fpc_enable_{prop}', 
               BoolProperty(name=prop.replace('_', ' ').title(), 
               default=default))
    
    bpy.app.timers.register(process_data)

def unregister():
    stop_receiving()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.udp_ip
    del bpy.types.Scene.udp_port
    del bpy.types.Scene.fpc_receiving
    del bpy.types.Scene.fpc_active_armature
    del bpy.types.Scene.fpc_debug_show
    del bpy.types.Scene.fpc_debug_data
    
    # 删除启用属性
    control_props = ['mouth', 'teeth', 'head', 
                   'left_eyelid', 'right_eyelid',
                   'left_brow', 'right_brow',
                   'left_pupil', 'right_pupil']
    
    for prop in control_props:
        if f'fpc_enable_{prop}' in bpy.types.Scene.__annotations__:
            del bpy.types.Scene.__annotations__[f'fpc_enable_{prop}']

if __name__ == "__main__":
    register()
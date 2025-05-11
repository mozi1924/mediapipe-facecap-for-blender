import bpy
import socket
import json
import threading
import queue
from mathutils import Vector
from bpy.types import Operator, Panel
from bpy.props import StringProperty, IntProperty, BoolProperty, PointerProperty

# 全局配置
UDP_IP = '127.0.0.1'
UDP_PORT = 12345
sock = None
is_receiving = False
udp_thread = None
data_queue = queue.Queue()
DEBUG_MAX_LINES = 20
BASE_ARMATURE_NAME = "FaceCapture_Rig"

controls = {
    'mouth':        'Mouth',
    'left_eyelid':  'LeftEyelid',
    'right_eyelid': 'RightEyelid',
    'left_pupil':   'LeftPupil',
    'right_pupil':  'RightPupil',
    'left_brow':    'LeftBrow',
    'right_brow':   'RightBrow',
    'head':         'Head',
    'teeth':        'Teeth'
}

PUPIL_MOVE_RANGE = 0.1

def get_armature(context, create_new=False):
    """获取或创建骨架（支持自动命名）"""
    if create_new:
        # 创建新骨架并进入编辑模式
        armature_data = bpy.data.armatures.new(BASE_ARMATURE_NAME)
        armature_obj = bpy.data.objects.new(BASE_ARMATURE_NAME, armature_data)
        context.collection.objects.link(armature_obj)
        
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 创建根骨骼
        root_bone = armature_data.edit_bones.new("Root")
        root_bone.head = Vector((0, 0, 0))
        root_bone.tail = Vector((0, 0, 0.2))
        
        # 创建控制骨骼
        for index, (control_type, bone_name) in enumerate(controls.items()):
            new_bone = armature_data.edit_bones.new(bone_name)
            x_offset = index * 0.3
            new_bone.head = root_bone.head + Vector((x_offset, 0, 0.1))
            new_bone.tail = new_bone.head + Vector((0, 0.2, 0))
            new_bone.parent = root_bone
        
        bpy.ops.object.mode_set(mode='OBJECT')
        return armature_obj
    
    # 返回当前选择的骨架
    return context.scene.fpc_active_armature

def get_pose_bone(armature, bone_name):
    """获取姿态骨骼"""
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
            
            # 1. 嘴巴控制
            mouth = get_pose_bone(armature, controls['mouth'])
            if mouth:
                mouth.scale[0] = 1.0 + info.get('mouth_width', 0.0)
                mouth.scale[2] = info.get('mouth_open', 0.0)
                if auto_key:
                    mouth.keyframe_insert(data_path='scale', frame=frame)

            # 2. 眼皮控制
            for side in ('left', 'right'):
                eyelid = get_pose_bone(armature, controls[f'{side}_eyelid'])
                if eyelid:
                    eyelid.scale[2] = info.get(f'{side}_eyelid', 0.0)
                    if auto_key:
                        eyelid.keyframe_insert(data_path='scale', frame=frame)

            # 3. 瞳孔控制
            for side in ('left', 'right'):
                pupil = get_pose_bone(armature, controls[f'{side}_pupil'])
                if pupil:
                    x = max(min(info.get(f'{side}_pupil_x', 0.0), PUPIL_MOVE_RANGE), -PUPIL_MOVE_RANGE)
                    y = max(min(info.get(f'{side}_pupil_y', 0.0), PUPIL_MOVE_RANGE), -PUPIL_MOVE_RANGE)
                    pupil.location = (x, y, 0)
                    if auto_key:
                        pupil.keyframe_insert(data_path='location', frame=frame)

            # 4. 眉毛控制
            for side in ('left', 'right'):
                brow = get_pose_bone(armature, controls[f'{side}_brow'])
                if brow:
                    brow.scale[2] = 1.0 + info.get(f'{side}_brow', 0.0)
                    if auto_key:
                        brow.keyframe_insert(data_path='scale', frame=frame)

            # 5. 头部控制
            head = get_pose_bone(armature, controls['head'])
            if head:
                head.rotation_euler = (
                    info.get('head_pitch', 0.0),
                    info.get('head_yaw', 0.0),
                    info.get('head_roll', 0.0)
                )
                if auto_key:
                    head.keyframe_insert(data_path='rotation_euler', frame=frame)

            # 6. 牙齿控制
            teeth = get_pose_bone(armature, controls['teeth'])
            if teeth:
                teeth.scale[2] = info.get('teeth_open', 0.0)
                if auto_key:
                    teeth.keyframe_insert(data_path='scale', frame=frame)

        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
    
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
            print(f"接收错误: {str(e)}")

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
    bl_label = "创建新面部控制骨架"

    def execute(self, context):
        new_armature = get_armature(context, create_new=True)
        context.scene.fpc_active_armature = new_armature
        self.report({'INFO'}, f"已创建骨架 {new_armature.name}")
        return {'FINISHED'}

class FPC_OT_Start(bpy.types.Operator):
    bl_idname = "fpc.start"
    bl_label = "开始接收数据"

    def execute(self, context):
        sc = context.scene
        start_receiving(sc.udp_ip, sc.udp_port)
        sc.fpc_receiving = True
        self.report({'INFO'}, "已启动 UDP 接收")
        return {'FINISHED'}

class FPC_OT_Stop(bpy.types.Operator):
    bl_idname = "fpc.stop"
    bl_label = "停止接收"

    def execute(self, context):
        stop_receiving()
        context.scene.fpc_receiving = False
        self.report({'INFO'}, "已停止 UDP 接收")
        return {'FINISHED'}

class FPC_PT_Panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FaceCapture'
    bl_label = '面部捕捉'

    def draw(self, context):
        sc = context.scene
        layout = self.layout
        
        # 骨架选择
        layout.prop_search(sc, "fpc_active_armature", sc, "objects", 
                          text="当前骨架", icon='ARMATURE_DATA')
        layout.operator('fpc.create_controls', icon='ADD')
        
        # UDP控制
        layout.separator()
        layout.prop(sc, 'udp_ip')
        layout.prop(sc, 'udp_port')
        
        if sc.fpc_receiving:
            layout.operator('fpc.stop', icon='CANCEL')
        else:
            layout.operator('fpc.start', icon='PLAY')

class FPC_PT_DebugPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FaceCapture'
    bl_label = "调试信息"
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
                box.label(text="等待数据...", icon='INFO')

classes = (
    FPC_OT_CreateControls,
    FPC_OT_Start,
    FPC_OT_Stop,
    FPC_PT_Panel,
    FPC_PT_DebugPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.udp_ip = StringProperty(
        name="UDP IP", default=UDP_IP)
    bpy.types.Scene.udp_port = IntProperty(
        name="UDP 端口", default=UDP_PORT, min=1, max=65535)
    bpy.types.Scene.fpc_receiving = BoolProperty(default=False)
    bpy.types.Scene.fpc_active_armature = PointerProperty(
        name="Active Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    bpy.types.Scene.fpc_debug_show = BoolProperty(
        name="显示调试信息",
        default=False
    )
    bpy.types.Scene.fpc_debug_data = StringProperty(
        name="调试数据",
        default=""
    )
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

if __name__ == "__main__":
    register()
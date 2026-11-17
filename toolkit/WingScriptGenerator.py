'''
Update date: 2026-5-8
Author: Zhe Wen

2026-04-28 修改calculate_wing_mac的返回内容
2026-04-01 取消脚本填写功能，全面转向使用GGS，仅保留参数映射功能
2026-04-01 新增机翼面积计算功能
2025-04-11 修正全局到局部的角度变换local_2_global
2026-05-08 修改加载加载superwing数据集加载
'''
from jinja2 import Template
from typing import Dict, List, Union, Optional, Any
from math import tan
from cst_modeling.section import cst_foil, cst_foil_fit
from pathlib import Path
import json
import numpy as np
import math

def get_effective_angles(v_prime):
    # 假设 v_prime 是一个 [x, y, z] 的 np.array
    # 确保向量已归一化
    v_n = v_prime / np.linalg.norm(v_prime)
    
    # 解算角度 (弧度)
    dihedral_new = np.arctan2(v_n[2], v_n[1])
    sweep_new = np.arctan2(v_n[0], v_n[1])
    
    return np.degrees(sweep_new), np.degrees(dihedral_new)

def local_2_global(loc_params: Dict, actual_root_twist, target_dihedral):
    #构造向量
    Lambda = np.radians(loc_params.get('sweep',0))
    if abs(Lambda)<=1e-2:
        # 后掠角很小的时候，直接不用计算，新的后掠角就是0,新的上反角就是目标上反角
        final_sweep = 0
    else:
        epsilon = np.radians(actual_root_twist)
        Gamma = np.radians(target_dihedral)
        final_sweep = np.degrees( np.arctan2(math.tan(Lambda)*math.cos(epsilon), math.cos(Gamma)) )
    loc_params["sweep"] = final_sweep
    loc_params["dihedral"] = target_dihedral

    #修改上反角
    L_physical = loc_params["span"] * np.cos(np.radians(target_dihedral)) 
    loc_params["span"] = L_physical
    loc_params["twist_root"] = actual_root_twist
    return loc_params

def global_2_local(global_params:Dict, actual_root_twist, target_root_twist=0, target_dihedral=0):
    '''
    将扭转角产生的影响进行映射，修改由其导致的后掠角、上反角变化.
    global_params: 包含全局参数的字典
    actual_root_twist: 该部件安装到飞机上后的实际根部扭转角
    target_root_twist: 该部件设计时的根部扭转角
    '''
    twist_change = target_root_twist - actual_root_twist
    origin_sweep = global_params["sweep"]
    origin_dihedral = global_params["dihedral"]
    # 计算新的后掠角和上反角
    rotatetion_matrix_twist = np.array([
        [np.cos(np.radians(twist_change)), 0, np.sin(np.radians(twist_change))],
        [0, 1, 0],
        [-np.sin(np.radians(twist_change)), 0, np.cos(np.radians(twist_change))]
    ])
    v = np.array([
        np.sin(np.radians(origin_sweep)) * np.cos(np.radians(origin_dihedral)),
        np.cos(np.radians(origin_sweep)) * np.cos(np.radians(origin_dihedral)),
        np.sin(np.radians(origin_dihedral))
    ])
    v_prime = rotatetion_matrix_twist @ v
    new_sweep, new_dihedral = get_effective_angles(v_prime)
    # print(v_prime)
    # 接下来做上反角的映射,涉及到span和sweep的变化
    dihedral_change = new_dihedral - target_dihedral
    rotatetion_matrix_dihedral = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(dihedral_change)), np.sin(np.radians(dihedral_change))],
        [0, -np.sin(np.radians(dihedral_change)), np.cos(np.radians(dihedral_change))]
    ])
    v_prime_prime = rotatetion_matrix_dihedral @ v_prime
    new_sweep, new_dihedral = get_effective_angles(v_prime_prime)
    global_params["sweep"] = new_sweep
    # global_params["dihedral"] = new_dihedral
    span_val = global_params.get("span", global_params.get("winglet_height", 0.0))
    global_params["span"] = span_val / np.cos(np.radians(dihedral_change)) 
    # print(new_sweep, new_dihedral)

    return global_params
    
def generate_cst_arrays(cst_u, cst_l, tcroot, thickness_scale, camber_scale):
    # print(f"厚度缩放: {thickness_scale:.6f}, 弯度缩放: {camber_scale:.6f}")
    # 计算新的厚度
    actual_thickness = tcroot * thickness_scale

    # 生成带厚度变化的翼型
    x_new, yu_thickness, yl_thickness, _, _ = cst_foil(
        1001, cst_u, cst_l, t=actual_thickness, tail=0
    )

    # 计算中弧线和厚度分布
    camber_line = 0.5 * (yu_thickness + yl_thickness)
    thickness_dist = yu_thickness - yl_thickness

    # 应用弯度变化
    # 这里假设弯度变化是通过调整中弧线实现的
    new_camber_line = camber_line * camber_scale

    # 重新构建上下表面
    yu_final = new_camber_line + 0.5 * thickness_dist
    yl_final = new_camber_line - 0.5 * thickness_dist

    # 拟合新的CST参数
    cst_u_new, cst_l_new = cst_foil_fit(
        x_new, yu_final, x_new, yl_final, n_cst=10, xn1=0.5, xn2=1.0
    )
    return cst_u_new, cst_l_new, actual_thickness
    
def generate_super_wing(config, padding=False):
    SA = config['SA']
    DA = config['DA']
    DA_kink = config['DA_kink']
    AR = config['AR']
    TR = config['TR']
    kink = config['kink']
    rootadj = config['rootadj']
    rtcs_0 = config['rtcs_0']
    rtcs_1 = config['rtcs_1']
    rtcs_2 = config['rtcs_2']
    cambers_0 = config['cambers_0']
    cambers_1 = config['cambers_1']
    cambers_2 = config['cambers_2']
    twists_0 = config['twists_0']
    twists_1 = config['twists_1']
    twists_2 = config['twists_2']
    twists_3 = config['twists_3']
    tcroot = config['tcroot']
    AF_upper = [config[f'U{i}'] for i in range(10)]
    AF_lower = [config[f'L{i}'] for i in range(10)]

    #Global calculations
    root_chord = 1.0
    tip_chord = root_chord * TR
    span = AR * (1 + TR) * 1/2
    ref_area = (1 + TR) * span * 1/4
    kink_chord = TR * kink + 1.0* (1 - kink)

    #Sec1
    xLE_kink = tan( config['SA'] * 3.141592653589793 / 180.0 ) * (span * kink)
    root_adj_l = rootadj * ( xLE_kink + kink_chord - 1.0 )
    adjusted_root_chord = 1.0 + root_adj_l
    # span1 = span * kink/2.0
    # span2 = span * kink
    upper_cst_params_root, lower_cst_params_root, _ = generate_cst_arrays(
        AF_upper, AF_lower, tcroot, 1.0, 0.0)
    upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
        AF_upper, AF_lower, tcroot, 1.0, config['cambers_0'] * config['cambers_1'])

    init_sec_params = {
        "span": span * kink/2.0,
        "chord_root": adjusted_root_chord,
        "chord_tip":kink_chord + (adjusted_root_chord-kink_chord)/2,
        "twist_root": 0,
        "twist_tip": twists_0,
        "sweep": SA,
        "dihedral": DA_kink,
        "upper_cst_params_root": upper_cst_params_root.tolist(),
        "lower_cst_params_root": lower_cst_params_root.tolist(),
        "upper_cst_params_tip": upper_cst_params_tip.tolist(),
        "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }
    
    #==== Ending of start section, and begin medium section ====#
    mid_sec_params = []
    #Sec2 
    if padding:
        upper_cst_params_root, lower_cst_params_root = upper_cst_params_tip, lower_cst_params_tip
        upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
            AF_upper, AF_lower, tcroot, config['rtcs_0'], config['cambers_1'])
        sec_params = {
            "span": span * kink/2.0,
            "chord_tip": kink_chord,
            "twist_tip": twists_0 + twists_1,
            "sweep": SA,
            "dihedral": DA_kink,
            "smooth_sector":0.0,
            "upper_cst_params_root": upper_cst_params_root.tolist(),
            "lower_cst_params_root": lower_cst_params_root.tolist(),
            "upper_cst_params_tip": upper_cst_params_tip.tolist(),
            "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }
    else:
        upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
            AF_upper, AF_lower, tcroot, config['rtcs_0'], config['cambers_1'])
        sec_params = {
            "span": span * kink/2.0,
            "chord_tip": kink_chord,
            "twist_tip": twists_0 + twists_1,
            "sweep": SA,
            "dihedral": DA_kink,
            "smooth_sector":0.0,
            "upper_cst_params_tip": upper_cst_params_tip.tolist(),
            "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }

    mid_sec_params.append(sec_params)

    #Sec3
    if padding:
        upper_cst_params_root, lower_cst_params_root = upper_cst_params_tip, lower_cst_params_tip
        upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
            AF_upper, AF_lower, tcroot, config['rtcs_0'] * config['rtcs_1'], 1)
        sec_params = {
            "span": span * (1.0 - kink)/2.0,
            "chord_tip": tip_chord + (kink_chord-tip_chord)/2,
            "twist_tip": twists_0 + twists_1 + twists_2,
            "sweep": SA,
            "dihedral": DA,
            "smooth_sector":0.0,
            "upper_cst_params_root": upper_cst_params_root.tolist(),
            "lower_cst_params_root": lower_cst_params_root.tolist(),
            "upper_cst_params_tip": upper_cst_params_tip.tolist(),
            "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }
    else:
        upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
            AF_upper, AF_lower, tcroot, config['rtcs_0'] * config['rtcs_1'], 1)
        sec_params = {
            "span": span * (1.0 - kink)/2.0,
            "chord_tip": tip_chord + (kink_chord-tip_chord)/2,
            "twist_tip": twists_0 + twists_1 + twists_2,
            "sweep": SA,
            "dihedral": DA,
            "smooth_sector":0.0,
            "upper_cst_params_tip": upper_cst_params_tip.tolist(),
            "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }

    mid_sec_params.append(sec_params)

    #Sec4
    if padding:
        upper_cst_params_root, lower_cst_params_root = upper_cst_params_tip, lower_cst_params_tip
        upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
            AF_upper, AF_lower, tcroot, config['rtcs_0'] * config['rtcs_1'] * config['rtcs_2'], config['cambers_2'])
        sec_params = {
            "span": span * (1.0 - kink)/2.0,
            "chord_tip": tip_chord,
            "twist_tip": twists_0 + twists_1 + twists_2 + twists_3,
            "sweep": SA,
            "dihedral": DA,
            "smooth_sector":0.0,
            "upper_cst_params_root": upper_cst_params_root.tolist(),
            "lower_cst_params_root": lower_cst_params_root.tolist(),
            "upper_cst_params_tip": upper_cst_params_tip.tolist(),
            "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }
    else:
        upper_cst_params_tip, lower_cst_params_tip, _ = generate_cst_arrays(
            AF_upper, AF_lower, tcroot, config['rtcs_0'] * config['rtcs_1'] * config['rtcs_2'], config['cambers_2'])
        sec_params = {
            "span": span * (1.0 - kink)/2.0,
            "chord_tip": tip_chord,
            "twist_tip": twists_0 + twists_1 + twists_2 + twists_3,
            "sweep": SA,
            "dihedral": DA,
            "smooth_sector":0.0,
            "upper_cst_params_tip": upper_cst_params_tip.tolist(),
            "lower_cst_params_tip": lower_cst_params_tip.tolist()
        }
    mid_sec_params.append(sec_params)

    #==== Winglet Section ====#
    winglet_sec_params = None

    return init_sec_params, mid_sec_params, winglet_sec_params

def superwing_dataset_converter(config, padding=False):
    init_sec_params, mid_sec_params, winglet_sec_params = generate_super_wing(config, padding=padding)
    init_sec_params, mid_sec_params, winglet_sec_params = batch_convert(init_sec_params, mid_sec_params, winglet_sec_params)
    ac_dict = {'Cond_0':[config['M30'], config['AA0']],
        'Cond_1':[config['M31'], config['AA1']],
        'Cond_2':[config['M32'], config['AA2']],
        'Cond_3':[config['M33'], config['AA3']],
        'Cond_4':[config['M34'], config['AA4']],
        'Cond_5':[config['M35'], config['AA5']],
        'Cond_6':[config['M36'], config['AA6']],
        'Cond_7':[config['M37'], config['AA7']],
                }
    return init_sec_params, mid_sec_params, winglet_sec_params, ac_dict

def batch_convert(init_sec_params: Dict[str, Union[float, int, List[float]]], 
                    mid_sec_params: List[Dict[str, Union[float, int]]], 
                    winglet_sec_params: Optional[Dict[str, Union[float, int, List[float]]]]):
    init_sec_params = convert_numpy_to_list(init_sec_params)
    mid_sec_params = convert_numpy_to_list(mid_sec_params)
    winglet_sec_params = convert_numpy_to_list(winglet_sec_params)
    return init_sec_params, mid_sec_params, winglet_sec_params

def read_configs_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    headers = lines[0].strip().split()
    configs = []
    for idx, line in enumerate(lines[1:], start=0):
        if line.strip():
            values = line.strip().split()
            config_dict = {'ID': idx}
            for header, val in zip(headers, values):
                config_dict[header] = float(val)
            configs.append(config_dict)
    return headers, configs

class DataManagement:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        # 1. 检查是否为 .json 文件
        if self.file_path.suffix.lower() != '.json':
            raise ValueError(f"File must be a JSON file, got: {file_path}")
        # 2. 确保父目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # 3. 如果文件不存在，创建一个空的 JSON 对象文件
        if not self.file_path.exists():
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        # 4. 加载当前数据到内存
        self._load()
    
    def _load(self):
        """从文件加载 JSON 数据到内存"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("JSON file must contain a top-level object (dict), not array or primitive.")
                self._data = data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {self.file_path}: {e}")
    
    def _save(self):
        """将内存中的数据写回文件"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)
       
    # ———————— CRUD 操作 ————————
    def add_entry(self, ID:int, 
                        init_sec_params: Dict[str, Union[float, int, List[float]]], 
                        mid_sec_params: List[Dict[str, Union[float, int]]], 
                        winglet_sec_params: Optional[Dict[str, Union[float, int, List[float]]]],
                        aerodynamic_char: Optional[Dict]=None) -> None:
        """新增条目（如果 ID 已存在则报错）"""
        entry_id = str(ID)
        if entry_id in self._data:
            raise KeyError(f"Entry with ID '{entry_id}' already exists.")
        self._data[entry_id] = {'ID':ID,
                          "wings":[
                              {"init":init_sec_params,
                                "mid":mid_sec_params,
                                "winglet":winglet_sec_params
                              }
                          ],
                          'aerodynamic_char':aerodynamic_char
                          }

    def upsert_entry(self, ID:int, 
                        init_sec_params: Dict[str, Union[float, int, List[float]]], 
                        mid_sec_params: List[Dict[str, Union[float, int]]], 
                        winglet_sec_params: Optional[Dict[str, Union[float, int, List[float]]]],
                        aerodynamic_char: Optional[Dict]=None) -> None:
        """插入或更新条目（存在则覆盖，不存在则新增）"""
        entry_id = str(ID)
        self._data[entry_id] = {'ID':ID,
                          "wings":[
                              {"init":init_sec_params,
                                "mid":mid_sec_params,
                                "winglet":winglet_sec_params
                              }
                          ],
                          'aerodynamic_char':aerodynamic_char
                          }

    def update_entry(self, ID: int, key:str, value: Dict[str, Any]) -> None:
        """更新条目中指定的字段（如果条目或字段不存在则报错）"""
        entry_id = str(ID)
        if entry_id not in self._data:
            raise KeyError(f"Entry with ID '{entry_id}' does not exist.")
        # 可选：检查 key 是否合法（比如只能是预定义的几个字段）
        allowed_keys = {'init_sec_params', 'mid_sec_params', 'winglet_sec_params', 'aerodynamic_char'}
        if key not in allowed_keys:
            raise ValueError(f"Invalid field name '{key}'. Allowed: {allowed_keys}")
        # 只更新该字段
        self._data[entry_id][key] = value


    def delete_entry(self, ID: int) -> None:
        entry_id = str(ID)
        """删除条目（如果不存在则报错）"""
        if entry_id not in self._data:
            raise KeyError(f"Entry with ID '{entry_id}' does not exist.")
        del self._data[entry_id]

    def get_entry(self, ID: int):
        entry_id = str(ID)
        """获取条目（不存在返回 None）"""
        return self._data.get(entry_id)

    def get_entry_full(self, ID: int):
        entry_id = str(ID)
        entry = self._data.get(entry_id)
        init_sec_params = entry['wings'][0]['init']
        mid_sec_params = entry['wings'][0]['mid']
        winglet_sec_params = entry['wings'][0]['winglet']
        aerodynamic_char = entry['aerodynamic_char']
        return init_sec_params, mid_sec_params, winglet_sec_params, aerodynamic_char

    def list_entries(self) -> Dict[str, Any]:
        """返回所有条目（浅拷贝）"""
        return self._data.copy()

    # ———————— 上下文管理器支持 ————————
    def save(self) -> None:
        """手动保存到磁盘"""
        self._save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 只有在没有异常时才保存？或者总是保存？
        # 这里选择：无论是否异常都保存（你可根据需求调整）
        self._save()

# ———————— 辅助函数 ————————
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # np.int64, np.float32 等
        return obj.item()  # 转为 Python int / float
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


'''
把GGS格式的参数分解为按照部件划分的参数字典，方便后续使用
'''
def twist_washout_convert(sec_params:Dict):
    twist_root = sec_params.get("twist_root")
    twist_tip = sec_params.get("twist_tip")
    if twist_root is not None and twist_tip is not None:
        sec_params["washout"] = twist_tip - twist_root
    # 处理只有 tip 有值的情况（如果逻辑需要）
    elif twist_tip is not None:
         # 这里根据你的业务逻辑处理，比如假设 root 为 0
         sec_params["washout"] = twist_tip 
     # 处理只有 root 有值的情况
    elif twist_root is not None:
         sec_params["washout"] = 0 # 或者其他默认逻辑
    sec_params.pop("twist_root", None), sec_params.pop("twist_tip", None)# 洗掉原来的扭转角参数，换成washout参数
    return sec_params

def ggs_wing_to_param_dict(ggs_wing_params: Dict[str, Any]):
    '''
    ggs_wing_params: 从GGS中提取的机翼参数字典，包含init_sec_params, mid_sec_params, winglet_sec_params等键，然后逐个处理
    输入示例：config["wings"]
    '''
    init_sec_params = ggs_wing_params["init"] #这是个字典
    mid_sec_params = ggs_wing_params['mid']   #这是个列表，里面每个元素都是一个字典，代表一个中间段
    winglet_sec_params = ggs_wing_params.get('winglet') #旧版的GGS有winglet参数，但新版的GGS可能没有，如果没有则返回None

    # 对初始段进行处理
    # 先用init段的参数得到mid的root处迭代项
    upper_cst_params_root = init_sec_params["upper_cst_params_tip"]
    lower_cst_params_root = init_sec_params["lower_cst_params_tip"]
    twist_root = init_sec_params["twist_tip"]
    chord_root = init_sec_params["chord_tip"]

    init_sec_params = global_2_local(init_sec_params, actual_root_twist=init_sec_params['twist_root'], target_root_twist=0)
    init_sec_params = twist_washout_convert(init_sec_params)

    # 对中间段进行处理
    
    for sec_params in mid_sec_params:
        if sec_params.get("span", 0)<=1e-2:
            continue
        # 先把迭代项写入
        sec_params["upper_cst_params_root"] = upper_cst_params_root
        sec_params["lower_cst_params_root"] = lower_cst_params_root
        sec_params["twist_root"] = twist_root
        sec_params["chord_root"] = chord_root
        # 再把下一段使用的迭代项提取出来
        upper_cst_params_root = sec_params["upper_cst_params_tip"]
        lower_cst_params_root = sec_params["lower_cst_params_tip"]
        twist_root = sec_params["twist_tip"]
        chord_root = sec_params["chord_tip"]
        # 通过坐标变化，把全局参数变成可描述的局部几何信息
        sec_params = global_2_local(sec_params, actual_root_twist=sec_params['twist_root'], target_root_twist=0)
        sec_params = twist_washout_convert(sec_params)
    
    # 对winglet进行处理
    if (winglet_sec_params is not None) and (winglet_sec_params.get('winglet_height',0)>1e-2):
        # 写入迭代项
        winglet_sec_params["upper_cst_params_root"] = upper_cst_params_root
        winglet_sec_params["lower_cst_params_root"] = lower_cst_params_root
        winglet_sec_params["twist_root"] = twist_root
        winglet_sec_params["chord_root"] = chord_root
        # 把height变为span
        winglet_sec_params["span"] = winglet_sec_params["winglet_height"]
        winglet_sec_params.pop("winglet_height", None)
        # 不需要提取迭代项，因为没有下一段了。直接把全局信息变为局部几何信息
        winglet_sec_params = global_2_local(winglet_sec_params, actual_root_twist=winglet_sec_params['twist_root'], target_root_twist=0)
        winglet_sec_params = twist_washout_convert(winglet_sec_params)
        # 根据新版GGS规范，没有winglet类，直接并入mid的最后一个元素
        mid_sec_params.append(winglet_sec_params)

    return init_sec_params, mid_sec_params

def param_to_ggs_wing_dict(geom_params:Dict, angle_of_incidence:float=0.0): #这里以后要考虑怎么把上反角加进来
    '''
    把按照部件划分的参数字典转换回GGS格式的机翼参数字典，方便后续使用
    输入示例：init_sec_params, mid_sec_params = ggs_wing_to_param_dict(config["wings"][0])
    输出示例：ggs_wing_dict = param_to_ggs_wing_dict(init_sec_params, mid_sec_params)
    '''
    # 先把局部几何信息转换回全局参数
    try:
        init_sec_params = geom_params['init']
        mid_sec_params = geom_params.get('mid', [])
    except:
        init_sec_params = geom_params
        mid_sec_params = geom_params.get('mid', [])

    # 先对初段进行处理，把局部参数转换回全局参数
    init_sec_params = local_2_global(init_sec_params, actual_root_twist=angle_of_incidence, target_dihedral=init_sec_params.get("dihedral", 0.0))
    # 映射回扭转角参数
    if "washout" in init_sec_params:
        init_sec_params["twist_tip"] = init_sec_params["washout"] + angle_of_incidence
        init_sec_params["twist_root"] = angle_of_incidence
        # print(f"根部扭转角: {init_sec_params['twist_root']:.2f}°, 翼尖扭转角: {init_sec_params['twist_tip']:.2f}°, 洗出角: {init_sec_params['washout']:.2f}°")
        init_sec_params.pop("washout", None)
    # 为了保证平滑，我们需要对转换后的后掠角、上反角、进行平滑处理，过滤掉三角变换引入的噪声
    last_sweep, last_dihedral = init_sec_params["sweep"], init_sec_params["dihedral"]
    # 计算出下一个站位的实际根部扭转角，作为迭代项传递给下一个站位
    last_twist_tip = init_sec_params["twist_tip"]

    for sec_params in mid_sec_params:
        sec_params = local_2_global(sec_params, actual_root_twist=last_twist_tip, target_dihedral=sec_params.get("dihedral", 0.0))
        if "washout" in sec_params:
            sec_params["twist_tip"] = sec_params["washout"] + last_twist_tip
            sec_params["twist_root"] = last_twist_tip
            sec_params.pop("washout", None)
        if abs(sec_params["sweep"] - last_sweep) <= 3e-1:
            sec_params["sweep"] = last_sweep
        if abs(sec_params["dihedral"] - last_dihedral) <= 3e-1:
            sec_params["dihedral"] = last_dihedral
        
        # 更新迭代项
        last_sweep, last_dihedral = sec_params["sweep"], sec_params["dihedral"]
        last_twist_tip = sec_params["twist_tip"]

    # 最后把局部参数重新组合成GGS格式的机翼参数字典
    ggs_wing_dict = {
        "init": init_sec_params,
        "mid": mid_sec_params
    }
    return ggs_wing_dict


def calculate_wing_area(wing_params: Dict[str, Any]) -> float:
    '''
    根据机翼参数计算机翼面积，输入参数格式与ggs_wing_to_param_dict函数的输出相同
    '''
    init_sec_params = wing_params['init']
    mid_sec_params = wing_params.get('mid', [])
    winglet_sec_params = wing_params.get('winglet')

    # 计算初始段面积
    area = 0.5 * (init_sec_params['chord_root'] + init_sec_params['chord_tip']) * init_sec_params['span']
    last_sec_params = init_sec_params

    # 计算中间段面积
    for sec in mid_sec_params:
        root_chord = sec.get('chord_root', last_sec_params['chord_tip']) # 如果中间段没有根部弦长，默认使用前一段的翼尖弦长
        area += 0.5 * (root_chord + sec['chord_tip']) * sec['span']
        last_sec_params = sec

    # 计算翼尖段面积（如果存在）
    if winglet_sec_params is not None:
        root_chord = winglet_sec_params.get('chord_root', last_sec_params['chord_tip'])
        area += 0.5 * (root_chord + winglet_sec_params['chord_tip']) * winglet_sec_params['winglet_height']
    return area


def calculate_wing_mac(wing_params: Dict[str, Any]) -> Dict[str, float]:
    '''
    根据机翼参数计算机翼面积、平均气动弦长(MAC)及其前缘位置(LEMAC)。
    假设输入包含各段的 sweep_le (前缘后掠角，弧度)。
    '''
    sections = []
    
    # 1. 提取所有段并建立统一列表
    # 初始段
    init = wing_params['init']
    init_sec_loc = wing_params.get('global', {}).get("x_rel_location", 0.0)
    sections.append({
        'cr': init['chord_root'],
        'ct': init['chord_tip'],
        'b': init['span'],
        'x_le': init_sec_loc, # 该段根部前缘的x坐标
        'sweep': init.get('sweep', 0.0), # 前缘后掠角
        'dihedral': init.get('dihedral', 0.0) # 上反角
    })
    
    # 中间段
    last_ct = init['chord_tip']
    last_x_tip = init_sec_loc + init['span'] * math.tan( math.radians(init.get('sweep', 0.0)) )
    
    for sec in wing_params.get('mid', []):
        cr = sec.get('chord_root', last_ct)
        sections.append({
            'cr': cr,
            'ct': sec['chord_tip'],
            'b': sec['span'],
            'x_le': last_x_tip,
            'sweep': sec.get('sweep', 0.0),
            'dihedral': sec.get('dihedral', 0.0)
        })
        last_ct = sec['chord_tip']
        last_x_tip += sec['span'] * math.tan(math.radians(sec.get('sweep', 0.0)))

    # 翼尖/翼梢小翼 (这里简化处理，将其视为一个水平段计算面积贡献)
    if wing_params.get('winglet'):
        w = wing_params['winglet']
        sections.append({
            'cr': w.get('chord_root', last_ct),
            'ct': w.get('chord_tip', 0),
            'b': w.get('winglet_height', 0.0),
            'x_le': last_x_tip,
            'sweep': w.get('sweep', 0.0),
            'dihedral': w.get('dihedral', 0.0)
        })

    # 2. 核心计算：面积加权
    total_area = 0.0
    weighted_mac = 0.0
    weighted_x_lemac = 0.0
    total_span = 0

    for s in sections:
        cr, ct, b = s['cr'], s['ct'], s['b']
        if cr + ct == 0: continue # 防止除以0
        
        # 当前段面积
        s_area = 0.5 * (cr + ct) * b * math.cos(math.radians(s['dihedral'])) # 考虑上反角对投影面积的影响
        
        # 当前段 MAC 长度
        s_mac = (2/3) * (cr + ct - (cr * ct) / (cr + ct))
        
        # 当前段 MAC 前缘相对于该段根部的纵向偏移
        # 偏移 = (b/3) * (cr + 2*ct) / (cr + ct) * tan(sweep)
        
        s_x_offset = (b/3) * ((cr + 2*ct) / (cr + ct)) * math.tan(math.radians(s['sweep']))  * math.cos(math.radians(s['dihedral'])) # 考虑上反角对前缘位置的影响
        s_x_lemac = s['x_le'] + s_x_offset
        
        # 累加
        total_area += s_area
        weighted_mac += s_mac * s_area
        weighted_x_lemac += s_x_lemac * s_area
        total_span += b * math.cos(math.radians(s['dihedral'])) # 累加投影后的展长

    # 3. 计算最终加权值
    mac_length = weighted_mac / total_area if total_area > 0 else 0
    x_lemac = weighted_x_lemac / total_area if total_area > 0 else 0

    return {
        "area": total_area,
        "mac_length": mac_length,
        "x_lemac": x_lemac,
        "span" : total_span
    }

def calculate_full_wing_aerodynamic_char(wing_params_list: List[Dict[str, Any]]) -> Dict[str, float]:
    '''
    对整机的机翼参数列表进行综合计算，得到整机的总翼面积、平均气动弦长(MAC)及其前缘位置(LEMAC)。
    '''
    total_area = 0.0
    weighted_mac = 0.0
    weighted_x_lemac = 0.0
    max_area = 0.0
    for wing_params in wing_params_list:
        x_rotation = wing_params.get('global', {}).get('x_rel_rotation', 0.0)
        wing_char = calculate_wing_mac(wing_params)

        effective_area = wing_char['area'] * math.cos(math.radians(x_rotation)) # 考虑机翼相对于水平面的旋转对有效面积的影响
        total_area += effective_area
        weighted_mac += wing_char['mac_length'] * effective_area
        weighted_x_lemac += wing_char['x_lemac'] * effective_area

        if wing_char['area'] > max_area:
            max_area = wing_char['area']
            main_wing_char = wing_char
    overall_mac = weighted_mac / total_area if total_area > 0 else 0
    overall_x_lemac = weighted_x_lemac / total_area if total_area > 0 else 0
    return {
        "total_area": total_area,
        "overall_mac": overall_mac,
        "overall_x_lemac": overall_x_lemac
    }, main_wing_char